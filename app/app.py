
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import streamlit as st
import torch
from huggingface_hub import snapshot_download

from config import MAX_PROMPT_TOKENS, MAX_SUMMARY_TOKENS
from models import load_tokenizer, load_causal_lm
from data import clean_post
from train_reward import load_reward_model

HF_REPO_ID = "Musfirah-umar/Rlhf-summarizer"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = ROOT / "checkpoints"
SFT_DIR = CKPT / "sft"
REWARD_DIR = CKPT / "reward"
PPO_DIR = CKPT / "ppo"
DPO_DIR = CKPT / "dpo"

METHODS = [
    ("SFT", SFT_DIR, "Supervised fine-tuned baseline"),
    ("PPO", PPO_DIR, "RLHF with a learned reward model"),
    ("DPO", DPO_DIR, "Direct preference optimisation"),
]

EXAMPLE_POST = """SUBREDDIT: r/relationships
TITLE: My roommate keeps eating my food and I don't know how to bring it up
POST: I (23F) have been living with my roommate (24F) for about eight months. We get along well and she is generally great, but over the last two months a lot of my groceries have been disappearing. Yogurt, leftovers, even the expensive cheese I bought for a dinner I was hosting. I labelled my shelf in the fridge and it made no difference. I don't want to turn this into a fight because the lease runs for another four months and I like her as a person. I've thought about just buying a mini fridge for my room, but that feels like admitting defeat and costs money I'd rather not spend. How do I raise this without making things awkward?"""


def ensure_checkpoints():
    if not (SFT_DIR / "model.safetensors").exists():
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=str(SFT_DIR),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )

    required = {
        "reward": REWARD_DIR / "reward_model.pt",
        "ppo": PPO_DIR / "model.safetensors",
        "dpo": DPO_DIR / "model.safetensors",
    }

    for folder, file in required.items():
        if not file.exists():
            snapshot_download(
                repo_id=HF_REPO_ID,
                local_dir=str(CKPT),
                allow_patterns=[f"{folder}/*"],
            )


@st.cache_resource(show_spinner="Loading models...")
def load_everything():
    ensure_checkpoints()

    tok = load_tokenizer(str(SFT_DIR))
    policies = {}

    for name, path, _ in METHODS:
        model = load_causal_lm(str(path)).float().to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        policies[name] = model

    reward_model = load_reward_model(str(REWARD_DIR), DEVICE).float().eval()

    return tok, policies, reward_model


def build_prompt_ids(tok, raw_text):
    post = raw_text.split("TL;DR:")[0].strip()
    marker_len = len(tok("\nTL;DR:", add_special_tokens=False)["input_ids"])
    budget = MAX_PROMPT_TOKENS - marker_len - 2

    post_ids = tok(post, add_special_tokens=False)["input_ids"]
    truncated = len(post_ids) > budget

    if truncated:
        post = tok.decode(post_ids[:budget])

    prompt_ids = tok(clean_post(post), add_special_tokens=False)["input_ids"]

    while len(prompt_ids) > MAX_PROMPT_TOKENS:
        post = tok.decode(tok(post, add_special_tokens=False)["input_ids"][:-4])
        prompt_ids = tok(clean_post(post), add_special_tokens=False)["input_ids"]
        truncated = True

    return prompt_ids, truncated


@torch.no_grad()
def summarize(model, tok, prompt_ids):
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    mask = torch.ones_like(ids)

    output = model.generate(
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=MAX_SUMMARY_TOKENS,
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    summary_ids = output[0, ids.size(1):].tolist()

    if tok.eos_token_id in summary_ids:
        summary_ids = summary_ids[:summary_ids.index(tok.eos_token_id) + 1]

    return summary_ids


@torch.no_grad()
def reward_score(reward_model, prompt_ids, summary_ids):
    if not summary_ids:
        return float("nan")

    ids = torch.tensor(
        [prompt_ids + summary_ids],
        dtype=torch.long,
        device=DEVICE
    )

    return reward_model(ids, torch.ones_like(ids)).item()


def run_all(raw_text):
    prompt_ids, truncated = build_prompt_ids(tok, raw_text)
    results = {}

    for name, _, _ in METHODS:
        summary_ids = summarize(policies[name], tok, prompt_ids)

        results[name] = {
            "text": tok.decode(summary_ids, skip_special_tokens=True).strip(),
            "reward": reward_score(reward_model, prompt_ids, summary_ids),
            "tokens": len(summary_ids),
        }

    return results, truncated


st.set_page_config(
    page_title="RLHF Summarizer",
    page_icon="📝",
    layout="wide"
)

st.title("RLHF Summarizer: SFT vs PPO vs DPO")
st.caption(
    "Compare summaries from SFT, PPO, and DPO with their reward scores. "
    "Base model: SmolLM2-135M."
)

with st.sidebar:
    st.header("About")
    st.write(f"Maximum summary length: {MAX_SUMMARY_TOKENS} tokens")
    st.write(f"Maximum prompt length: {MAX_PROMPT_TOKENS} tokens")
    st.write("Higher reward scores indicate a more preferred summary.")
    st.write(f"Device: {DEVICE}")

tok, policies, reward_model = load_everything()
st.success("Models loaded. Paste a post below.")

if "post" not in st.session_state:
    st.session_state["post"] = ""

if st.button("Load an example post"):
    st.session_state["post"] = EXAMPLE_POST

post = st.text_area(
    "Reddit Post",
    key="post",
    height=260,
    placeholder="Paste your Reddit post here..."
)

if st.button("Summarize", type="primary"):
    if not post.strip():
        st.warning("Paste a post first.")
    else:
        with st.spinner("Generating summaries..."):
            results, truncated = run_all(post)

        if truncated:
            st.info(
                f"The post was shortened to fit the {MAX_PROMPT_TOKENS}-token limit."
            )

        valid = {
            name: data["reward"]
            for name, data in results.items()
            if data["reward"] == data["reward"]
        }

        best = max(valid, key=valid.get) if valid else None
        cols = st.columns(3)

        for col, (name, _, description) in zip(cols, METHODS):
            result = results[name]

            with col:
                st.subheader(name + (" (Best)" if name == best else ""))
                st.caption(description)
                st.metric(
                    "Reward Score",
                    f"{result['reward']:.3f}"
                    if result["reward"] == result["reward"]
                    else "N/A"
                )
                st.markdown("**Summary**")
                st.info(result["text"] or "(Empty summary)")
                st.caption(f"{result['tokens']} tokens")

        if best:
            st.caption(f"Highest reward score: {best}")
