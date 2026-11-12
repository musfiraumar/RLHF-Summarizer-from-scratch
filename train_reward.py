import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_preference_splits
from starter.batching import collate_pref
from starter.config import SEED


BATCH_SIZE = 8
LR = 1e-5


class RewardModel(nn.Module):
    def __init__(self, sft_path="checkpoints/sft"):
        super().__init__()

        self.body = load_causal_lm(sft_path).model
        self.body.config.use_cache = False

        hidden_size = self.body.config.hidden_size

        self.head = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        hidden = self.body(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False
        ).last_hidden_state

        last_idx = attention_mask.sum(dim=1) - 1

        batch_idx = torch.arange(
            input_ids.size(0),
            device=input_ids.device
        )

        last_hidden = hidden[batch_idx, last_idx]

        return self.head(last_hidden).squeeze(-1)


def reward_loss(r_w, r_l):
    loss = -F.logsigmoid(r_w - r_l).mean()
    accuracy = (r_w > r_l).float().mean()

    return loss, accuracy


def evaluate_reward(model, loader, device):
    was_training = model.training
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) for k, v in batch.items()
            }

            r_w = model(batch["chosen_ids"], batch["chosen_mask"])
            r_l = model(batch["rejected_ids"], batch["rejected_mask"])

            correct += (r_w > r_l).sum().item()
            total += len(r_w)

    model.train(was_training)

    return correct / total


def load_reward_model(path="checkpoints/reward", device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = RewardModel()

    state = torch.load(
        Path(path) / "reward_model.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return model


def save_mistakes(model, val_pairs, tok, device):
    mistakes = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(val_pairs), BATCH_SIZE):
            pairs = val_pairs[start:start + BATCH_SIZE]

            batch = collate_pref(pairs, tok.pad_token_id)

            batch = {
                k: v.to(device) for k, v in batch.items()
            }

            r_w = model(batch["chosen_ids"], batch["chosen_mask"])
            r_l = model(batch["rejected_ids"], batch["rejected_mask"])

            for i in range(len(pairs)):
                if r_w[i] <= r_l[i]:
                    mistakes.append({
                        "prompt": pairs[i]["prompt"],
                        "chosen": pairs[i]["chosen"],
                        "rejected": pairs[i]["rejected"],
                        "r_w": r_w[i].item(),
                        "r_l": r_l[i].item()
                    })

                if len(mistakes) >= 20:
                    break

            if len(mistakes) >= 20:
                break

    Path("results").mkdir(exist_ok=True)

    with open("results/reward_mistakes.json", "w", encoding="utf-8") as f:
        json.dump(mistakes, f, indent=4, ensure_ascii=False)

    print("Saved mistakes:", len(mistakes))


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print("Device:", device)

    tok = load_tokenizer()
    train_data, val_data = load_preference_splits(tok)

    print("Training pairs:", len(train_data))
    print("Validation pairs:", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: collate_pref(x, tok.pad_token_id)
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: collate_pref(x, tok.pad_token_id)
    )

    model = RewardModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    baseline_acc = evaluate_reward(
        model,
        val_loader,
        device
    )

    print("Baseline validation accuracy:", baseline_acc)

    steps = []
    train_losses = []

    val_steps = [0]
    val_accs = [baseline_acc]

    running_loss = 0.0
    start_time = time.time()

    model.train()

    for step, batch in enumerate(tqdm(train_loader), start=1):
        batch = {
            k: v.to(device) for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        r_w = model(
            batch["chosen_ids"],
            batch["chosen_mask"]
        )

        r_l = model(
            batch["rejected_ids"],
            batch["rejected_mask"]
        )

        loss, accuracy = reward_loss(r_w, r_l)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % 50 == 0:
            avg_loss = running_loss / 50
            running_loss = 0.0

            steps.append(step)
            train_losses.append(avg_loss)

            print(
                f"Step {step} | Train Loss: {avg_loss:.4f}"
            )

        if step % 250 == 0:
            val_acc = evaluate_reward(
                model,
                val_loader,
                device
            )

            val_steps.append(step)
            val_accs.append(val_acc)

            print(
                f"Step {step} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Validation Accuracy: {val_acc:.4f}"
            )

    total_time = time.time() - start_time

    final_acc = evaluate_reward(
        model,
        val_loader,
        device
    )

    final_train_loss = (
        train_losses[-1] if train_losses else running_loss / max(len(train_loader), 1)
    )

    if val_steps[-1] != step:
        val_steps.append(step)
        val_accs.append(final_acc)

    print("Final validation accuracy:", final_acc)
    print("Training time:", round(total_time, 2), "seconds")

    save_dir = Path("checkpoints/reward")
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        save_dir / "reward_model.pt"
    )

    tok.save_pretrained(save_dir)
    print("Reward model saved.")
    Path("results").mkdir(exist_ok=True)

    with open("results/reward_log.json", "w") as f:
        json.dump({
            "train_steps": steps,
            "train_losses": train_losses,
            "validation_steps": val_steps,
            "validation_accuracies": val_accs,
            "baseline_validation_accuracy": baseline_acc,
            "final_validation_accuracy": final_acc,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "training_time": total_time,
            "gpu": gpu
        }, f, indent=4)

    table_path = Path("results/tables/table2_training.md")
    table_path.parent.mkdir(parents=True, exist_ok=True)

    row = (
        f"| Reward Model | {final_train_loss:.4f} | "
        f"{final_acc:.4f} accuracy | "
        f"{total_time:.1f}s / {gpu} |"
    )

    if table_path.exists():
        lines = table_path.read_text(encoding="utf-8").splitlines()

        lines = [
            line for line in lines
            if not line.startswith("| Reward Model |")
        ]

        lines.append(row)
    else:
        lines = [
            "| Stage | Final train loss | Validation metric | Time / GPU |",
            "|---|---:|---:|---|",
            row
        ]

    table_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    plt.plot(
        steps,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        val_steps,
        val_accs,
        marker="o",
        label="Validation Accuracy"
    )
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title("Reward Model Training")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/figure2_reward_training.png"
    )

    plt.close()
    save_mistakes(
        model,
        val_data,
        tok,
        device
    )

    print("Reward model training completed.")
if __name__ == "__main__":
    main()