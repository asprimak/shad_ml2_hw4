import nltk
import pandas as pd
import streamlit as st
import torch
from transformers import pipeline

MODEL_PATH = "aprimak/claimbuster_bert"
LABELS = {
    "NFS": {"color": "#a8d5ba", "description": "Non-factual Sentence"},
    "UFS": {"color": "#f7dc6f", "description": "Unimportant Factual Sentence"},
    "CFS": {"color": "#f1948a", "description": "Check-worthy Factual Sentence"},
    "   ": {"color": "#e0e0e0", "description": "None (model's confidence < 0.7"},
}
EXAMPLES_CSV = "examples.csv"


@st.cache_data
def load_examples():
    return pd.read_csv(EXAMPLES_CSV)


@st.cache_resource
def load_model():
    nltk.download("punkt_tab", quiet=True)
    return pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        top_k=3,  # return scores for all 3 classes
        device="cpu",
        torch_dtype=torch.float32,
    )


def split_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


def classify(sents: list[str], clf) -> list[dict]:
    results = clf(sents, batch_size=32)
    rows = []
    for sent, preds in zip(sents, results):
        best = max(preds, key=lambda x: x["score"])
        rows.append(
            {
                "sentence": sent,
                "label": best["label"],
                "confidence": best["score"],
                "scores": {p["label"]: round(p["score"], 3) for p in preds},
            }
        )
    return rows


def render_highlighted(rows: list[dict], threshold: float) -> str:
    spans = []
    for r in rows:
        color = (
            LABELS[r["label"]]["color"] if r["confidence"] >= threshold else "#e0e0e0"
        )
        spans.append(
            f'<span style="background:{color};padding:2px 4px;border-radius:3px;'
            f'margin:1px;display:inline" title="{r["label"]} ({r["confidence"]:.0%})">'
            f"{r['sentence']}</span> "
        )
    return "".join(spans)


st.set_page_config(page_title="Claim Highlighter", layout="wide")
st.title("🔍 Claim Highlighter")
st.caption("Enter political text and see which sentences contain check-worthy claims.")

with st.sidebar:
    st.markdown("### Legend")
    for label, info in LABELS.items():
        st.markdown(
            f'<span style="background:{info["color"]};padding:4px 10px;border-radius:4px">'
            f"**{label}**</span> - {info['description']}",
            unsafe_allow_html=True,
        )

examples = load_examples()
col_left, col_right = st.columns(2)
if col_left.button("Random Clinton"):
    sample = examples[examples["speaker"] == "CLINTON"].sample(1).iloc[0]
    st.session_state["input_text"] = sample["text"]
if col_right.button("Random Trump"):
    sample = examples[examples["speaker"] == "TRUMP"].sample(1).iloc[0]
    st.session_state["input_text"] = sample["text"]


text = st.text_area(
    "Enter text",
    value=st.session_state.get("input_text", ""),
    height=180,
    placeholder="Politilcal speech or debate transcript",
)

if text.strip():
    clf = load_model()
    sents = split_sentences(text)
    rows = classify(sents, clf)

    st.subheader("Highlighted text")
    st.markdown(render_highlighted(rows, 0.7), unsafe_allow_html=True)

    with st.expander("Detailed predictions"):
        df = pd.DataFrame(
            [
                {
                    "Sentence": r["sentence"],
                    "Prediction": r["label"],
                    "Confidence": f"{r['confidence']:.1%}",
                    **r["scores"],
                }
                for r in rows
            ]
        )
        st.dataframe(df, use_container_width=True)
