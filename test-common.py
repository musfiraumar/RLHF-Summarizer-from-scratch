import torch
import sys
sys.path.insert(0, "starter")
from common import token_logprobs, summary_mask
from functools import partial
from torch.utils.data import DataLoader
from starter.models import load_tokenizer, load_causal_lm
from starter.data import load_sft_splits
from starter.batching import collate_sft
tok = load_tokenizer()
sft_train, sft_val = load_sft_splits(tok)

loader = DataLoader(
    sft_train,
    batch_size=8,
    shuffle=True,
    collate_fn=partial(collate_sft, pad_id=tok.pad_token_id)
)

batch = next(iter(loader))

mask = summary_mask(batch["prompt_lens"], batch["attention_mask"])

decoded = batch["input_ids"][0, 1:][mask[0].bool()]

print("Decoded summary:")
print(tok.decode(decoded))

model = load_causal_lm()

device = next(model.parameters()).device

logprobs = token_logprobs(
    model,
    batch["input_ids"].to(device),
    batch["attention_mask"].to(device)
)
print("Log probabilities shape:", logprobs.shape)
print("Summary mask shape:", mask.shape)