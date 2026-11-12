import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "starter"))

import os
import json
import time
import torch
import matplotlib.pyplot as plt

from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from starter.config import SEED
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits, load_eval_posts
from starter.batching import collate_sft, collate_prompts
from common import token_logprobs, summary_mask, generate_summaries


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)
eval_posts = load_eval_posts(tok)

model = load_causal_lm().to(device)
model.gradient_checkpointing_enable()
model.config.use_cache = False

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/summaries", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# DataLoaders
train_loader = DataLoader(
    sft_train,
    batch_size=16,
    shuffle=True,
    collate_fn=partial(collate_sft, pad_id=tok.pad_token_id)
)
val_loader = DataLoader(
    sft_val,
    batch_size=16,
    shuffle=False,
    collate_fn=partial(collate_sft, pad_id=tok.pad_token_id)
)

# Generate summaries before training
eval_batch = collate_prompts(
    eval_posts[:5],
    pad_id=tok.pad_token_id
)
base_summaries = generate_summaries(model, tok, eval_batch)

# SFT Loss
def sft_loss(model, batch):
    lp = token_logprobs(model, batch["input_ids"], batch["attention_mask"])
    mask = summary_mask(batch["prompt_lens"], batch["attention_mask"]).float()
    return -(lp * mask).sum() / mask.sum()

# Validation
def evaluate_val(model, val_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            lp = token_logprobs(model, batch["input_ids"], batch["attention_mask"])
            mask = summary_mask(batch["prompt_lens"], batch["attention_mask"]).float()
            total_loss += -(lp * mask).sum().item()
            total_tokens += mask.sum().item()
    model.train()
    return total_loss / total_tokens

# Baseline validation loss
baseline_val_loss = evaluate_val(model, val_loader)
print("Baseline validation loss:", baseline_val_loss)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
steps = []
train_losses = []
val_steps = [0]
val_losses = [baseline_val_loss]
running_loss = 0
total_steps = len(train_loader)
start_time = time.time()
model.train()

for step, batch in enumerate(tqdm(train_loader), start=1):
    batch = {
        key: value.to(device) for key, value in batch.items()
    }

    optimizer.zero_grad(set_to_none=True)

    full_mask = summary_mask(
        batch["prompt_lens"],
        batch["attention_mask"]
    )

    full_token_count = full_mask.sum().float()
    half = batch["input_ids"].size(0) // 2
    first_batch = {
        key: value[:half] for key, value in batch.items()
    }

    second_batch = {
        key: value[half:] for key, value in batch.items()
    }
    batch_loss = 0

    for micro_batch in [first_batch, second_batch]:
        lp = token_logprobs(
            model,
            micro_batch["input_ids"],
            micro_batch["attention_mask"]
        )
        mask = summary_mask(
            micro_batch["prompt_lens"],
            micro_batch["attention_mask"]
        ).float()

        loss_sum = -(lp * mask).sum()
        micro_loss = loss_sum / full_token_count
        micro_loss.backward()
        batch_loss += micro_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    running_loss += batch_loss

    if step % 50 == 0:
        avg_train_loss = running_loss / 50
        running_loss = 0
        steps.append(step)
        train_losses.append(avg_train_loss)
        print(
            f"Step {step}/{total_steps} | "
            f"Train Loss: {avg_train_loss:.4f}"
        )
    if step % 250 == 0:
        val_loss = evaluate_val(model, val_loader)
        val_steps.append(step)
        val_losses.append(val_loss)
        print(
            f"Step {step}/{total_steps} | "
            f"Validation Loss: {val_loss:.4f}"
        )
total_time = time.time() - start_time

# Save model and tokenizer
model.save_pretrained("checkpoints/sft")
tok.save_pretrained("checkpoints/sft")

# Save training logs
log_data = {
    "steps": steps,
    "train_losses": train_losses,
    "val_steps": val_steps,
    "val_losses": val_losses,
    "baseline_val_loss": baseline_val_loss,
    "final_val_loss": val_losses[-1],
    "total_training_time_seconds": total_time,
    "training_time_includes_validation": True,
    "gpu": gpu_name
}
with open("results/sft_log.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, indent=4)

# Plot training results
plt.figure(figsize=(8, 5))
plt.plot(steps, train_losses, label="Training Loss")
plt.plot(val_steps, val_losses, marker="o", label="Validation Loss")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("SFT Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/figures/figure1_sft_loss.png",dpi=300,bbox_inches="tight")
plt.close()

# Generate summaries after training
sft_summaries = generate_summaries(model, tok, eval_batch)
comparison = []
for i in range(5):
    comparison.append({
        "prompt": eval_posts[i]["prompt"],
        "reference_summary": eval_posts[i]["summary"],
        "before_sft": base_summaries[i],
        "after_sft": sft_summaries[i]
    })
with open("results/summaries/before_after_sft.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=4, ensure_ascii=False)

# Save Table 2
table2 = f"""# Table 2 – Training

| Stage | Final train loss | Validation metric | Time / GPU |
|---|---:|---:|---|
| SFT | {train_losses[-1]:.4f} | Val. loss: {val_losses[-1]:.4f} | {total_time / 60:.2f} min (including validation) / {gpu_name} |
"""

with open("results/tables/table2_training.md", "w", encoding="utf-8") as f:
    f.write(table2)


print("\nSFT training completed.")
print("Training time:", round(total_time / 60, 2), "minutes")
print("GPU:", gpu_name)
print("Baseline validation loss:", baseline_val_loss)
print("Final validation loss:", val_losses[-1])

print("\nBefore SFT summaries:")
for summary in base_summaries:
    print(summary)

print("\nAfter SFT summaries:")
for summary in sft_summaries:
    print(summary)
