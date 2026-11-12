import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from starter import config
from starter.data import load_sft_splits
from common import load_causal_lm, load_reward_model, token_logprobs, summary_mask


CLIP_EPS = 0.2
BETA_KL = 0.05
C_V = 0.1
C_H = 0.01
GAMMA = 1.0
PPO_EPOCHS = 4
LR = 1e-5
MAX_GRAD_NORM = 1.0
PROMPTS_PER_ROLLOUT = 16
ITERATIONS = 200
MAX_NEW_TOKENS = 48
N_HALVES = 2
PROMPT_MAX_LEN = 448
GRAD_CKPT = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFT_DIR = "checkpoints/sft"
PPO_DIR = "checkpoints/ppo"


def get_prompts(tok):
    train_data, _ = load_sft_splits(tok)
    return [item["prompt"] for item in train_data]

def collate_left(tok, prompts):
    tok.padding_side = "left"
    tok.truncation_side = "left"

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=PROMPT_MAX_LEN
    )

    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

class PolicyWithValue(nn.Module):
    def __init__(self, lm):
        super().__init__()

        self.lm = lm
        self.value_head = nn.Linear(lm.config.hidden_size, 1)

        nn.init.normal_(self.value_head.weight, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, ids, mask):
        hidden = self.lm.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False
        ).last_hidden_state

        logits = self.lm.lm_head(hidden)
        values = self.value_head(hidden.float()).squeeze(-1)

        return logits[:, :-1].float(), values[:, :-1].float()


def logp_and_entropy(logits, ids):
    logz = torch.logsumexp(logits, dim=-1)

    logp = logits.gather(
        -1,
        ids[:, 1:, None]
    ).squeeze(-1) - logz

    probs = torch.softmax(logits, dim=-1)
    entropy = logz - (probs * logits).sum(-1)

    return logp, entropy


@torch.no_grad()
def rollout(policy, ref, rm, tok, prompts):
    was_training = policy.training
    policy.eval()

    prompt_ids, prompt_mask = collate_left(tok, prompts)

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    generated = policy.lm.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        pad_token_id=pad_id,
        eos_token_id=eos_id
    )[:, prompt_ids.size(1):]

    is_eos = generated == eos_id

    gen_len = torch.where(
        is_eos.any(1),
        is_eos.float().argmax(1) + 1,
        torch.full_like(generated[:, 0], generated.size(1))
    )

    prompt_lens = prompt_mask.sum(1)

    sequences = [
        torch.cat([
            prompt_ids[i][prompt_mask[i].bool()],
            generated[i, :gen_len[i]]
        ])
        for i in range(generated.size(0))
    ]

    max_len = max(len(seq) for seq in sequences)

    ids = torch.full(
        (len(sequences), max_len),
        pad_id,
        dtype=torch.long,
        device=DEVICE
    )

    mask = torch.zeros_like(ids)

    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1

    summary = summary_mask(prompt_lens, mask).float()

    assert torch.equal(summary.sum(1).long(), gen_len)

    logits, old_values = policy(ids, mask)
    old_logp, _ = logp_and_entropy(logits, ids)

    ref_logp = token_logprobs(ref, ids, mask).float()
    scores = rm(ids, mask).float().view(-1)

    kl_tokens = (old_logp - ref_logp) * summary
    rewards = -BETA_KL * kl_tokens

    last = prompt_lens + gen_len - 2
    rewards[
        torch.arange(len(sequences), device=DEVICE),
        last
    ] += scores

    returns = torch.zeros_like(rewards)
    running = torch.zeros(len(sequences), device=DEVICE)

    for t in reversed(range(rewards.size(1))):
        running = (
            rewards[:, t] * summary[:, t]
            + GAMMA * running * summary[:, t]
        )
        returns[:, t] = running

    advantages = (returns - old_values) * summary
    selected = summary > 0

    advantages = torch.where(
        selected,
        (advantages - advantages[selected].mean())
        / (advantages[selected].std() + 1e-8),
        torch.zeros_like(advantages)
    )

    policy.train(was_training)

    stats = {
        "score": scores.mean().item(),
        "kl": kl_tokens.sum(1).mean().item(),
        "length": gen_len.float().mean().item()
    }

    batch = {
        "ids": ids,
        "mask": mask,
        "summary": summary,
        "old_logp": old_logp,
        "advantages": advantages,
        "returns": returns
    }

    samples = [
        (
            prompts[i][-150:],
            tok.decode(
                generated[i, :gen_len[i]],
                skip_special_tokens=True
            )
        )
        for i in range(min(3, len(prompts)))
    ]

    return batch, stats, samples

def ppo_update(policy, optimizer, batch):
    batch_size = batch["ids"].size(0)
    token_count = batch["summary"].sum()

    halves = torch.arange(batch_size, device=DEVICE).chunk(N_HALVES)

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
        "first_ratio_dev": 0.0
    }

    policy.train()

    for epoch in range(PPO_EPOCHS):
        optimizer.zero_grad(set_to_none=True)

        for indices in halves:
            length = int(batch["mask"][indices].sum(1).max())

            ids = batch["ids"][indices, :length]
            mask = batch["mask"][indices, :length]
            summary = batch["summary"][indices, :length - 1]
            old_logp = batch["old_logp"][indices, :length - 1]
            advantages = batch["advantages"][indices, :length - 1]
            returns = batch["returns"][indices, :length - 1]

            logits, values = policy(ids, mask)
            logp, entropy = logp_and_entropy(logits, ids)

            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(
                ratio,
                1 - CLIP_EPS,
                1 + CLIP_EPS
            )

            policy_loss = -(
                torch.min(
                    ratio * advantages,
                    clipped * advantages
                ) * summary
            ).sum() / token_count

            value_loss = (
                ((values - returns) ** 2) * summary
            ).sum() / token_count

            entropy_mean = (entropy * summary).sum() / token_count

            loss = (
                policy_loss
                + C_V * value_loss
                - C_H * entropy_mean
            )

            loss.backward()

            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item() / PPO_EPOCHS
                stats["value_loss"] += value_loss.item() / PPO_EPOCHS
                stats["entropy"] += entropy_mean.item() / PPO_EPOCHS

                clipped_tokens = (
                    (ratio - 1).abs() > CLIP_EPS
                ).float()

                stats["clip_fraction"] += (
                    (clipped_tokens * summary).sum().item()
                    / token_count.item()
                    / PPO_EPOCHS
                )

                if epoch == 0:
                    ratio_dev = (
                        (ratio - 1).abs() * summary
                    ).max().item()

                    stats["first_ratio_dev"] = max(
                        stats["first_ratio_dev"],
                        ratio_dev
                    )

        torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            MAX_GRAD_NORM
        )

        optimizer.step()

    return stats


def save_policy(policy, tok):
    os.makedirs(PPO_DIR, exist_ok=True)

    policy.lm.save_pretrained(PPO_DIR)
    tok.save_pretrained(PPO_DIR)


def save_log(log):
    os.makedirs("results", exist_ok=True)

    with open("results/ppo_log.json", "w") as f:
        json.dump(log, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    iterations = 3 if args.debug else ITERATIONS
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    tok = AutoTokenizer.from_pretrained(SFT_DIR)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = PolicyWithValue(
        load_causal_lm(SFT_DIR).to(DEVICE)
    ).to(DEVICE)

    if GRAD_CKPT:
        policy.lm.gradient_checkpointing_enable()

    policy.lm.config.use_cache = True
    ref = load_causal_lm(SFT_DIR).to(DEVICE).eval()
    for param in ref.parameters():
        param.requires_grad_(False)
    rm = load_reward_model().to(DEVICE).eval()

    for param in rm.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LR
    )

    prompts = get_prompts(tok)
    random.Random(config.SEED).shuffle(prompts)
    assert len(prompts) >= iterations * PROMPTS_PER_ROLLOUT
    log = []
    start_time = time.time()

    for iteration in range(iterations):
        start = iteration * PROMPTS_PER_ROLLOUT
        end = start + PROMPTS_PER_ROLLOUT
        batch_prompts = prompts[start:end]

        batch, rollout_stats, samples = rollout(
            policy,
            ref,
            rm,
            tok,
            batch_prompts
        )

        update_stats = ppo_update(policy, optimizer, batch)
        record = {
            "iter": iteration + 1,
            "reward": rollout_stats["score"],
            "kl": rollout_stats["kl"],
            "length": rollout_stats["length"],
            "clip_frac": update_stats["clip_fraction"],
            "policy_loss": update_stats["policy_loss"],
            "value_loss": update_stats["value_loss"],
            "entropy": update_stats["entropy"],
            "minutes": (time.time() - start_time) / 60
        }
        log.append(record)
        print(
            f"[{iteration + 1}/{iterations}] "
            f"reward={record['reward']:.3f} "
            f"kl={record['kl']:.3f} "
            f"length={record['length']:.1f} "
            f"clip={record['clip_frac']:.3f} "
            f"policy_loss={record['policy_loss']:.4f} "
            f"value_loss={record['value_loss']:.4f} "
            f"entropy={record['entropy']:.3f} "
            f"({record['minutes']:.1f} min)"
        )

        if iteration == 0:
            print(
                f"Check: KL={rollout_stats['kl']:.2e}, "
                f"ratio difference={update_stats['first_ratio_dev']:.2e}"
            )

        if (iteration + 1) % 10 == 0 or args.debug:
            for prompt, summary in samples[:2]:
                print(
                    "...",
                    prompt.replace("\n", " "),
                    "->",
                    summary
                )

        if (iteration + 1) % 50 == 0:
            save_policy(policy, tok)
            save_log(log)
    save_log(log)

    if not args.debug:
        save_policy(policy, tok)
        print("Saved policy to", PPO_DIR)

if __name__ == "__main__":
    main()