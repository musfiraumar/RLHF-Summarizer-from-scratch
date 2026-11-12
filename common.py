import torch
def token_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    selected = torch.gather(logits, 2, targets.unsqueeze(-1)).squeeze(-1)
    log_probs = selected-torch.logsumexp(logits, dim=-1)
    return log_probs

def summary_mask(prompt_lens, attention_mask):
    prompt_lens = prompt_lens.to(attention_mask.device)
    B, L = attention_mask.shape
    positions = torch.arange(L - 1, device=attention_mask.device)
    start = prompt_lens - 1
    end = attention_mask.sum(dim=1) - 2
    mask = (positions.unsqueeze(0) >= start.unsqueeze(1)) & \
           (positions.unsqueeze(0) <= end.unsqueeze(1))
    return mask.long()

def generate_summaries(model, tok, prompt_batch, max_new_tokens=48):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_batch["input_ids"].to(device)
    attention_mask = prompt_batch["attention_mask"].to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id
        )
    output = output[:, input_ids.shape[1]:]
    summaries = [
        tok.decode(ids, skip_special_tokens=True)
        for ids in output
    ]
    model.train(was_training)
    return summaries
