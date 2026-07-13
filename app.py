"""
app.py
-------
Streamlit dashboard for the Cyber Threat Detection System.

Features:
  - Single-message threat classification (paste a log/alert line)
  - Batch classification via CSV upload
  - Confidence scores per class
  - Summary visualizations (class distribution, confidence, trend)
  - Toggle between the fast TF-IDF baseline and the BERT model

Run with:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import clean_text  # noqa: E402

LABEL_NAMES = ["Benign", "Phishing", "Malware", "DDoS", "Intrusion", "SQL_Injection"]
SEVERITY_COLOR = {
    "Benign": "#16a34a",
    "Phishing": "#f59e0b",
    "Malware": "#dc2626",
    "DDoS": "#dc2626",
    "Intrusion": "#ea580c",
    "SQL_Injection": "#ea580c",
}

BASELINE_PATH = "models/baseline_tfidf_lr.joblib"
BERT_PATH = "models/bert_threat_model"


@st.cache_resource(show_spinner=False)
def load_baseline():
    import joblib
    if not os.path.exists(BASELINE_PATH):
        return None
    return joblib.load(BASELINE_PATH)


@st.cache_resource(show_spinner=False)
def load_bert():
    if not os.path.exists(BERT_PATH):
        return None
    try:
        from bert_model import BertThreatClassifier
        return BertThreatClassifier.load(BERT_PATH)
    except Exception as e:  # transformers/torch may not be installed
        st.warning(f"Could not load BERT model: {e}")
        return None


def predict_baseline(pipeline, texts):
    probs = pipeline.predict_proba(texts)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict_bert(clf, texts):
    probs = clf.predict_proba(texts)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def main():
    st.set_page_config(page_title="Cyber Threat Detection System", page_icon="🛡️", layout="wide")

    st.title("🛡️ Cyber Threat Detection System")
    st.caption("BERT-based classifier for triaging security logs & alerts into threat categories")

    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.radio(
            "Model",
            ["TF-IDF + Logistic Regression (baseline)", "BERT (fine-tuned)"],
            index=0,
            help="The BERT model requires torch + transformers and a trained checkpoint in models/bert_threat_model",
        )
        st.markdown("---")
        st.markdown(
            "**Classes detected:**\n\n"
            "- ✅ Benign\n"
            "- 🎣 Phishing\n"
            "- 🦠 Malware\n"
            "- 🌊 DDoS\n"
            "- 🔓 Intrusion / Brute-Force\n"
            "- 🧨 SQL Injection"
        )
        st.markdown("---")
        st.caption("Tech stack: Python · Scikit-learn · Transformers (BERT) · Pandas · Matplotlib · Streamlit")

    use_bert = model_choice.startswith("BERT")
    model = load_bert() if use_bert else load_baseline()

    if model is None:
        st.error(
            f"Model not found. Train it first with `python src/train.py` "
            f"(expects a checkpoint at `{BERT_PATH if use_bert else BASELINE_PATH}`)."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔎 Single Message", "📁 Batch (CSV) Analysis", "📊 Model Info"])

    # ---------------- Tab 1: single message ----------------
    with tab1:
        st.subheader("Classify a single log / alert message")
        example = "Multiple failed SSH login attempts detected on 192.168.1.14 from external IP 45.33.10.9"
        text_input = st.text_area("Log / alert text", value=example, height=100)

        if st.button("Analyze", type="primary"):
            cleaned = clean_text(text_input)
            if use_bert:
                pred, probs = predict_bert(model, [cleaned])
            else:
                pred, probs = predict_baseline(model, [cleaned])

            label = LABEL_NAMES[pred[0]]
            confidence = probs[0][pred[0]]
            color = SEVERITY_COLOR[label]

            st.markdown(
                f"### Prediction: <span style='color:{color}'>{label}</span> "
                f"({confidence:.1%} confidence)",
                unsafe_allow_html=True,
            )
            if label != "Benign":
                st.warning(f"⚠️ Potential **{label}** activity detected — recommend SOC review.")
            else:
                st.success("No malicious indicators detected.")

            fig, ax = plt.subplots(figsize=(7, 3))
            bars = ax.bar(LABEL_NAMES, probs[0], color=[SEVERITY_COLOR[l] for l in LABEL_NAMES])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.bar_label(bars, fmt="%.2f")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

    # ---------------- Tab 2: batch CSV ----------------
    with tab2:
        st.subheader("Batch-classify logs from a CSV file")
        st.caption("CSV must contain a `text` column with one log/alert message per row.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                df["text_clean"] = df["text"].astype(str).apply(clean_text)
                with st.spinner("Classifying..."):
                    if use_bert:
                        preds, probs = predict_bert(model, df["text_clean"].tolist())
                    else:
                        preds, probs = predict_baseline(model, df["text_clean"].tolist())

                df["predicted_label"] = [LABEL_NAMES[p] for p in preds]
                df["confidence"] = probs.max(axis=1)

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(
                        df[["text", "predicted_label", "confidence"]].style.format({"confidence": "{:.1%}"}),
                        use_container_width=True,
                        height=400,
                    )
                with col2:
                    counts = df["predicted_label"].value_counts()
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(
                        counts.values, labels=counts.index, autopct="%1.0f%%",
                        colors=[SEVERITY_COLOR[l] for l in counts.index],
                    )
                    ax.set_title("Threat Class Distribution")
                    st.pyplot(fig)

                    n_threats = (df["predicted_label"] != "Benign").sum()
                    st.metric("Flagged as malicious", f"{n_threats} / {len(df)}")

                st.download_button(
                    "⬇️ Download results as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "threat_predictions.csv",
                    "text/csv",
                )

    # ---------------- Tab 3: model info ----------------
    with tab3:
        st.subheader("Model information")
        st.markdown(f"**Active model:** {model_choice}")
        if use_bert:
            st.markdown(
                "- Architecture: `bert-base-uncased` fine-tuned with a sequence "
                "classification head (6 classes)\n"
                "- Framework: HuggingFace `transformers` + PyTorch\n"
                "- Max sequence length: 64 tokens"
            )
        else:
            st.markdown(
                "- Architecture: TF-IDF (1-2 grams, 20k features) + Logistic Regression\n"
                "- Framework: scikit-learn\n"
                "- Purpose: fast baseline / CPU-only fallback"
            )
        st.info(
            "Evaluation charts (confusion matrix, per-class F1, ROC curves) are generated by "
            "`src/evaluate.py` and saved to `outputs_demo/`."
        )


if __name__ == "__main__":
    main()
