from functools import partial
from torch.utils.data import DataLoader
from models import load_tokenizer
from data import load_sft_splits, load_preference_splits, load_eval_posts
from batching import collate_sft, collate_pref, collate_prompts


tok = load_tokenizer()

sft_train, sft_val = load_sft_splits(tok)
pref_train, pref_val = load_preference_splits(tok)
eval_posts = load_eval_posts(tok)

print("SFT train / val:", len(sft_train), len(sft_val))
print("Pref train / val:", len(pref_train), len(pref_val))
print("Eval posts:", len(eval_posts))


ex = sft_train[0]
print("\n--- one SFT example ---")
print(ex["prompt"][-300:])
print("SUMMARY:", repr(ex["summary"]))


pad = tok.pad_token_id

b = next(iter(DataLoader(
    sft_train,
    8,
    shuffle=True,
    collate_fn=partial(collate_sft, pad_id=pad)
)))

print("\nSFT batch:", tuple(b["input_ids"].shape),
      "prompt_lens", b["prompt_lens"].tolist())


b = next(iter(DataLoader(
    pref_train,
    8,
    shuffle=True,
    collate_fn=partial(collate_pref, pad_id=pad)
)))

print("Pref batch:", tuple(b["chosen_ids"].shape),
      tuple(b["rejected_ids"].shape))


b = next(iter(DataLoader(
    eval_posts,
    8,
    collate_fn=partial(collate_prompts, pad_id=pad)
)))

print("Prompt batch:", tuple(b["input_ids"].shape),
      "(left-padded)")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import summary_mask

# 1. Count check (first 8 examples, without shuffling)

batch = collate_sft(
    sft_train[:8],
    pad_id=pad
)

mask = summary_mask(
    batch["prompt_lens"],
    batch["attention_mask"]
)

print("\n--- Summary Mask Count Check ---")

print("Mask counts:", mask.sum(dim=1).tolist())

print("Actual summary lengths:", [
    len(x["summary_ids"]) for x in sft_train[:8]
])


# 2. SFT dataset statistics (all 20,000 examples)

prompt_lengths = [len(x["prompt_ids"]) for x in sft_train]
summary_lengths = [len(x["summary_ids"]) for x in sft_train]

print("\n--- SFT Dataset Statistics ---")

print("Prompt mean:", sum(prompt_lengths) / len(prompt_lengths))
print("Prompt max:", max(prompt_lengths))

print("Summary mean:", sum(summary_lengths) / len(summary_lengths))
print("Summary max:", max(summary_lengths))


import os

os.makedirs("results/tables", exist_ok=True)
table = f"""# Table 1 – Data
| Metric | Value |
|---|---:|
| SFT train / validation pairs | {len(sft_train):,} / {len(sft_val):,} |
| Preference train / validation pairs | {len(pref_train):,} / {len(pref_val):,} |
| Evaluation posts | {len(eval_posts):,} |
| Mean / max prompt length, SFT train (tokens) | {sum(prompt_lengths) / len(prompt_lengths):.2f} / {max(prompt_lengths)} |
| Mean / max summary length, SFT train (tokens) | {sum(summary_lengths) / len(summary_lengths):.2f} / {max(summary_lengths)} |
"""

with open("results/tables/table1_data.md", "w", encoding="utf-8") as f:
    f.write(table)

print("\nTable 1 saved to results/tables/table1_data.md")
