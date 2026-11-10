import torch
def token_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

def summary_mask(prompt_lens, attention_mask):
    B, L = attention_mask.shape
    positions = torch.arange(L - 1, device=attention_mask.device)
    start = prompt_lens - 1
    end = attention_mask.sum(dim=1) - 2
    mask = (positions.unsqueeze(0) >= start.unsqueeze(1)) & \
           (positions.unsqueeze(0) <= end.unsqueeze(1))
    return mask.long()

