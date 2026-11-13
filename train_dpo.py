import os
import sys
import json
import time
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from starter import config
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_preference_splits
from starter.batching import collate_pref
from common import token_logprobs, summary_mask


BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
LR = 2e-6
BETA = 0.1
MAX_GRAD_NORM = 1.0
TOTAL_STEPS = 2500
LOG_EVERY = 250

SEED = config.SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SFT_DIR = "checkpoints/sft"
DPO_DIR = "checkpoints/dpo"
PARTIAL_DIR = "checkpoints/dpo_partial"

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


tok = load_tokenizer(SFT_DIR)

train_data, val_data = load_preference_splits(tok)

print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))


train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: collate_pref(batch, tok.pad_token_id),
    drop_last=True
)

val_loader = DataLoader(
    val_data,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda batch: collate_pref(batch, tok.pad_token_id)
)


policy = load_causal_lm(SFT_DIR).float().to(DEVICE)
ref = load_causal_lm(SFT_DIR).float().to(DEVICE)

policy.config.use_cache = False
ref.config.use_cache = False

policy.gradient_checkpointing_enable()

for p in ref.parameters():
    p.requires_grad = False

ref.eval()

ref_trainable_params = sum(
    p.requires_grad for p in ref.parameters()
)

print("Policy dtype:", next(policy.parameters()).dtype)
print("Reference dtype:", next(ref.parameters()).dtype)
print("Trainable reference parameters:", ref_trainable_params)

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=LR
)


def get_margins(policy, ref, batch):

    chosen_ids = batch["chosen_ids"]
    chosen_mask = batch["chosen_mask"]

    rejected_ids = batch["rejected_ids"]
    rejected_mask = batch["rejected_mask"]

    prompt_lens = batch["prompt_lens"]

    chosen_summary_mask = summary_mask(
        prompt_lens,
        chosen_mask
    ).float()

    rejected_summary_mask = summary_mask(
        prompt_lens,
        rejected_mask
    ).float()

    policy_chosen_logps = token_logprobs(
        policy,
        chosen_ids,
        chosen_mask
    )

    policy_rejected_logps = token_logprobs(
        policy,
        rejected_ids,
        rejected_mask
    )

    with torch.no_grad():

        ref_chosen_logps = token_logprobs(
            ref,
            chosen_ids,
            chosen_mask
        )

        ref_rejected_logps = token_logprobs(
            ref,
            rejected_ids,
            rejected_mask
        )

    policy_chosen = (
        policy_chosen_logps * chosen_summary_mask
    ).sum(dim=1)

    policy_rejected = (
        policy_rejected_logps * rejected_summary_mask
    ).sum(dim=1)

    ref_chosen = (
        ref_chosen_logps * chosen_summary_mask
    ).sum(dim=1)

    ref_rejected = (
        ref_rejected_logps * rejected_summary_mask
    ).sum(dim=1)

    margin = BETA * (
        (policy_chosen - ref_chosen)
        - (policy_rejected - ref_rejected)
    )

    return margin


def evaluate(policy, ref, loader):

    policy.eval()
    ref.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            margin = get_margins(
                policy,
                ref,
                batch
            )

            correct += (margin > 0).sum().item()
            total += margin.numel()

    policy.train()

    return correct / max(total, 1)


first_batch = next(iter(val_loader))

first_batch = {
    key: value.to(DEVICE)
    for key, value in first_batch.items()
}

policy.eval()

with torch.no_grad():

    initial_margin = get_margins(
        policy,
        ref,
        first_batch
    )

    sanity_loss = (
        -F.logsigmoid(initial_margin).mean().item()
    )

policy.train()

print("Initial DPO loss:", sanity_loss)

assert abs(sanity_loss - np.log(2)) < 1e-4, (
    "Initial DPO loss should be approximately 0.6931"
)

baseline = evaluate(
    policy,
    ref,
    val_loader
)

print("Baseline validation accuracy:", baseline)


log = [{
    "step": 0,
    "train_loss": sanity_loss,
    "val_accuracy": baseline
}]


train_iterator = iter(train_loader)

running_loss = 0.0
start_time = time.time()

policy.train()

for step in range(1, TOTAL_STEPS + 1):

    try:
        batch = next(train_iterator)

    except StopIteration:
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

    batch = {
        key: value.to(DEVICE)
        for key, value in batch.items()
    }

    optimizer.zero_grad(set_to_none=True)

    batch_loss = 0.0

    for start in range(0, BATCH_SIZE, MICRO_BATCH_SIZE):

        end = start + MICRO_BATCH_SIZE

        micro_batch = {
            key: value[start:end]
            for key, value in batch.items()
        }

        margin = get_margins(
            policy,
            ref,
            micro_batch
        )

        loss = (
            -F.logsigmoid(margin).sum()
            / BATCH_SIZE
        )

        loss.backward()

        batch_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        MAX_GRAD_NORM
    )

    optimizer.step()

    running_loss += batch_loss

    if step % LOG_EVERY == 0:

        avg_train_loss = running_loss / LOG_EVERY

        val_accuracy = evaluate(
            policy,
            ref,
            val_loader
        )

        log.append({
            "step": step,
            "train_loss": avg_train_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Step {step} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        os.makedirs(PARTIAL_DIR, exist_ok=True)

        policy.save_pretrained(
            PARTIAL_DIR,
            safe_serialization=True
        )

        print("Partial checkpoint saved:", PARTIAL_DIR)

        running_loss = 0.0


training_time = time.time() - start_time

final_train_loss = log[-1]["train_loss"]
final_accuracy = log[-1]["val_accuracy"]

gpu_name = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)

print("\nDPO Training Complete")
print("Final train loss:", final_train_loss)
print("Final validation accuracy:", final_accuracy)
print("Training time:", round(training_time, 2), "seconds")
print("GPU:", gpu_name)


os.makedirs(DPO_DIR, exist_ok=True)

policy.save_pretrained(
    DPO_DIR,
    safe_serialization=True
)

tok.save_pretrained(DPO_DIR)

print("DPO model and tokenizer saved to:", DPO_DIR)


log_path = os.path.join(
    RESULTS_DIR,
    "dpo_log.json"
)

with open(log_path, "w") as f:
    json.dump(log, f, indent=4)


steps = [item["step"] for item in log]
losses = [item["train_loss"] for item in log]
accuracies = [item["val_accuracy"] for item in log]


plt.figure()
plt.plot(steps, losses, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("DPO Training Loss")
plt.title("DPO Training Loss")
plt.grid(True)
plt.savefig(
    os.path.join(FIGURES_DIR, "figure3_dpo_loss.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


plt.figure()
plt.plot(steps, accuracies, marker="o")
plt.xlabel("Training Steps")
plt.ylabel("Validation Accuracy")
plt.title("DPO Validation Accuracy")
plt.grid(True)
plt.savefig(
    os.path.join(FIGURES_DIR, "figure4_dpo_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()


readme_path = os.path.join(
    RESULTS_DIR,
    "dpo_readme.md"
)

with open(readme_path, "w") as f:

    f.write("# DPO Training Results\n\n")

    f.write("## Training Configuration\n\n")
    f.write(f"- Batch size: {BATCH_SIZE}\n")
    f.write(f"- Micro-batch size: {MICRO_BATCH_SIZE}\n")
    f.write(f"- Learning rate: {LR}\n")
    f.write(f"- Beta: {BETA}\n")
    f.write(f"- Total steps: {TOTAL_STEPS}\n")
    f.write("- Optimizer: AdamW\n")
    f.write(f"- Device: {gpu_name}\n\n")

    f.write("## Reference Model\n\n")
    f.write(
        f"- Trainable reference parameters: "
        f"{ref_trainable_params}\n\n"
    )

    f.write("## Results\n\n")
    f.write(f"- Initial DPO loss: {sanity_loss:.4f}\n")
    f.write(f"- Initial validation accuracy: {baseline:.4f}\n")
    f.write(f"- Final train loss: {final_train_loss:.4f}\n")
    f.write(f"- Final validation accuracy: {final_accuracy:.4f}\n")
    f.write(f"- Training time: {training_time:.2f} seconds\n")
    f.write(f"- GPU: {gpu_name}\n\n")

    f.write(
        "Training loss values are averages over each "
        "250-step logging interval. The final train loss "
        "is the average for steps 2251-2500.\n\n"
    )

    f.write("## Figures\n\n")
    f.write("- `figures/figure3_dpo_loss.png`\n")
    f.write("- `figures/figure4_dpo_accuracy.png`\n")


table_path = os.path.join(
    TABLES_DIR,
    "table2_training.md"
)

if os.path.exists(table_path):

    with open(table_path, "r") as f:
        table_text = f.read()

else:

    table_text = (
        "| Stage | Final train loss | Validation metric | Time / GPU |\n"
        "|---|---:|---:|---|\n"
    )

dpo_row = (
    f"| DPO | {final_train_loss:.4f} | "
    f"{final_accuracy:.4f} | "
    f"{training_time / 60:.2f} min / {gpu_name} |"
)

if "| DPO |" in table_text:

    lines = table_text.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("| DPO |"):
            lines[i] = dpo_row

    table_text = "\n".join(lines) + "\n"

else:

    table_text = table_text.rstrip() + "\n" + dpo_row + "\n"

with open(table_path, "w") as f:
    f.write(table_text)


print("\nResults saved:")
print(log_path)
print(readme_path)
print(table_path)
print(os.path.join(FIGURES_DIR, "figure3_dpo_loss.png"))
print(os.path.join(FIGURES_DIR, "figure4_dpo_accuracy.png"))