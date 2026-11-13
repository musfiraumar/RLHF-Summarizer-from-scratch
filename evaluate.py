
import os
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
from rouge_score import rouge_scorer
from huggingface_hub import snapshot_download

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_eval_posts
from starter.batching import collate_prompts
from common import token_logprobs, summary_mask
from train_reward import load_reward_model


HF_REPO_ID = "Musfirah-umar/Rlhf-summarizer"
BATCH_SIZE = 16
MAX_NEW_TOKENS = 48
N_EVAL_POSTS = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SFT_DIR = ROOT / "checkpoints/sft"
REWARD_DIR = ROOT / "checkpoints/reward"
PPO_DIR = ROOT / "checkpoints/ppo"
DPO_DIR = ROOT / "checkpoints/dpo"


def download_models():
    if not (SFT_DIR / "model.safetensors").exists():
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(SFT_DIR),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json"
            ]
        )

    for folder in ["reward", "ppo", "dpo"]:
        path = ROOT / "checkpoints" / folder

        if not path.exists():
            snapshot_download(
                repo_id=HF_REPO_ID,
                local_dir=str(ROOT / "checkpoints"),
                allow_patterns=[f"{folder}/*"]
            )


def generate_summaries(model, tok, posts):
    summaries = []
    model.eval()

    for start in range(0, len(posts), BATCH_SIZE):
        batch = posts[start:start + BATCH_SIZE]
        inputs = collate_prompts(batch, tok.pad_token_id)

        ids = inputs["input_ids"].to(DEVICE)
        mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            output = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        for row in output[:, ids.size(1):]:
            row = row.tolist()

            if tok.eos_token_id in row:
                row = row[:row.index(tok.eos_token_id) + 1]

            summaries.append(row)

    return summaries


def evaluate_model(model, reference, reward_model, posts, summaries, tok):
    rewards = []
    kl_scores = []

    for start in range(0, len(posts), BATCH_SIZE):
        batch_posts = posts[start:start + BATCH_SIZE]
        batch_summaries = summaries[start:start + BATCH_SIZE]

        sequences = [
            p["prompt_ids"] + s
            for p, s in zip(batch_posts, batch_summaries)
        ]

        max_len = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), max_len),
            tok.pad_token_id,
            dtype=torch.long
        )
        mask = torch.zeros_like(ids)

        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1

        ids = ids.to(DEVICE)
        mask = mask.to(DEVICE)

        prompt_lens = torch.tensor([
            len(p["prompt_ids"]) for p in batch_posts
        ], device=DEVICE)

        with torch.no_grad():
            rewards.extend(
                reward_model(ids, mask).cpu().tolist()
            )

            if model is reference:
                kl_scores.extend([0.0] * len(batch_posts))
            else:
                token_mask = summary_mask(prompt_lens, mask).float()

                method_log_probs = token_logprobs(model, ids, mask)
                sft_log_probs = token_logprobs(reference, ids, mask)

                kl = (
                    (method_log_probs - sft_log_probs) * token_mask
                ).sum(dim=1)

                kl_scores.extend(kl.cpu().tolist())

    return rewards, kl_scores


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    download_models()

    tok = load_tokenizer(str(SFT_DIR))
    posts = load_eval_posts(tok)[:N_EVAL_POSTS]

    if len(posts) != N_EVAL_POSTS:
        raise ValueError(f"Expected 300 posts, found {len(posts)}")

    reward_model = load_reward_model(
        str(REWARD_DIR), DEVICE
    ).float().eval()

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = {}

    for name, path in [
        ("sft", SFT_DIR),
        ("ppo", PPO_DIR),
        ("dpo", DPO_DIR)
    ]:
        print("Evaluating", name.upper())

        model = load_causal_lm(str(path)).float().to(DEVICE).eval()
        summaries = generate_summaries(model, tok, posts)

        if name == "sft":
            reference = model
        else:
            reference = load_causal_lm(str(SFT_DIR)).float().to(DEVICE).eval()

        rewards, kl = evaluate_model(
            model, reference, reward_model, posts, summaries, tok
        )

        results[name] = {
            "summaries": [
                tok.decode(s, skip_special_tokens=True).strip()
                for s in summaries
            ],
            "rewards": rewards,
            "kl": kl,
            "lengths": [len(s) for s in summaries],
            "rouge": [
                rouge.score(
                    p["summary"].strip(),
                    tok.decode(s, skip_special_tokens=True).strip()
                )["rougeL"].fmeasure
                for p, s in zip(posts, summaries)
            ]
        }

        if name != "sft":
            del reference

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    records = []

    for i, post in enumerate(posts):
        record = {
            "prompt": post["prompt"],
            "reference": post["summary"]
        }

        for name in ["sft", "ppo", "dpo"]:
            record[f"{name}_summary"] = results[name]["summaries"][i]
            record[f"{name}_reward"] = results[name]["rewards"][i]
            record[f"{name}_kl"] = results[name]["kl"][i]
            record[f"{name}_length"] = results[name]["lengths"][i]
            record[f"{name}_rougeL"] = results[name]["rouge"][i]

        record["ppo_win"] = (
            results["ppo"]["rewards"][i] > results["sft"]["rewards"][i]
        )
        record["dpo_win"] = (
            results["dpo"]["rewards"][i] > results["sft"]["rewards"][i]
        )

        records.append(record)

    os.makedirs(ROOT / "results", exist_ok=True)

    with open(ROOT / "results/eval_summaries.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for name, data in results.items():
        print(f"\n{name.upper()}")
        print("Mean reward:", round(np.mean(data["rewards"]), 4))
        print("Mean KL:", round(np.mean(data["kl"]), 4))
        print("Mean length:", round(np.mean(data["lengths"]), 2))
        print("Mean ROUGE-L:", round(np.mean(data["rouge"]), 4))

        if name != "sft":
            wins = sum(
                results[name]["rewards"][i] > results["sft"]["rewards"][i]
                for i in range(N_EVAL_POSTS)
            )
            print("Win rate:", round(wins / N_EVAL_POSTS, 4))

    print("\nSaved: results/eval_summaries.jsonl")


if __name__ == "__main__":
    main()
