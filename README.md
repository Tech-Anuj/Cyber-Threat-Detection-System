# 🛡️ Cyber Threat Detection System

A BERT-based text classifier that triages security logs and alerts (SIEM/IDS/firewall/email
messages) into threat categories — **Benign, Phishing, Malware, DDoS, Intrusion / Brute-Force,
SQL Injection** — with an interactive Streamlit dashboard for analysts.

**Tech stack:** Python · Scikit-learn · Transformers (BERT) · Pandas · Matplotlib · Streamlit

---

## 1. Project Structure

```
cyber_threat_detection/
├── app.py                       # Streamlit dashboard (main entry point)
├── requirements.txt
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator (swap for real SOC logs)
│   └── cyber_threats.csv        # Generated dataset (~2k labeled log messages)
├── src/
│   ├── preprocessing.py         # Text cleaning + stratified train/val/test split
│   ├── baseline_model.py        # TF-IDF + Logistic Regression (scikit-learn baseline)
│   ├── bert_model.py            # BERT fine-tuning wrapper (transformers + PyTorch)
│   ├── train.py                 # End-to-end training script (baseline + BERT)
│   └── evaluate.py              # Confusion matrix, F1, ROC curve generation (matplotlib)
├── models/                      # Saved model artifacts (created after training)
└── outputs_demo/                # Evaluation charts + reports (created after evaluate.py)
```

## 2. How It Works

1. **Data** — `generate_dataset.py` produces realistic, templated log/alert text across 6
   classes (e.g. *"Multiple failed SSH login attempts detected on 192.168.1.14 from external IP
   45.33.10.9"*). Swap this out for real data (CICIDS2017, Zeek/Suricata logs, SOC tickets) by
   pointing `preprocessing.load_and_split()` at your own CSV with `text` and `label` columns.
2. **Baseline model** — a TF-IDF + Logistic Regression pipeline (`scikit-learn`) trains in
   seconds and acts as a sanity check and CPU-only fallback.
3. **BERT model** — `bert-base-uncased` is fine-tuned with a sequence-classification head
   (`transformers` `Trainer` API) to capture contextual meaning that keyword/TF-IDF approaches
   miss (e.g. distinguishing a legitimate login from a credential-stuffing attempt described in
   similar wording).
4. **Evaluation** — `evaluate.py` produces a confusion matrix, per-class F1 bar chart, and
   one-vs-rest ROC curves with `matplotlib`, plus a scikit-learn classification report.
5. **Deployment** — `app.py` (Streamlit) lets an analyst paste a single alert or upload a CSV of
   logs for batch triage, with confidence scores and severity highlighting.

## 3. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `torch` + `transformers` are required only for the BERT model. If you only want the
> scikit-learn baseline (e.g. no GPU available), you can skip them and use `baseline_model.py`
> and the "TF-IDF" option in the Streamlit sidebar.

## 4. Usage

### Step 1 — Generate the dataset
```bash
cd data
python generate_dataset.py --n_samples 3000 --out cyber_threats.csv
```

### Step 2 — Train the models
```bash
cd ../src
python train.py --data ../data/cyber_threats.csv --epochs 4 --batch_size 16
```
This trains and saves:
- `../models/baseline_tfidf_lr.joblib`
- `../models/bert_threat_model/` (BERT weights + tokenizer)
- `../data/test_split.csv` (held-out test set for evaluation)

To train **only** the baseline (fast, no GPU needed):
```bash
python baseline_model.py
```

### Step 3 — Evaluate
```bash
python evaluate.py --model baseline --data ../data/test_split.csv
python evaluate.py --model bert --data ../data/test_split.csv
```
Outputs go to `../outputs_demo/`: `*_confusion_matrix.png`, `*_f1_per_class.png`,
`*_roc_curves.png`, `*_report.txt`.

### Step 4 — Run the dashboard
```bash
cd ..
streamlit run app.py
```
Open the local URL Streamlit prints (default `http://localhost:8501`). Use the sidebar to
switch between the BERT model and the TF-IDF baseline, paste a single log line, or upload a
CSV (must contain a `text` column) for batch analysis.

## 5. Extending to Production

- **Real data:** replace the synthetic generator with labeled data from your SIEM export,
  Suricata/Zeek `eve.json` logs, or a public IDS dataset (CICIDS2017, UNSW-NB15) — just map
  fields into `text` (a description/log line) and `label` (integer class).
- **Model choice:** swap `MODEL_NAME` in `bert_model.py` for `distilbert-base-uncased` (faster),
  `roberta-base` (often stronger), or a domain-pretrained checkpoint if available.
- **Class imbalance:** real-world benign traffic vastly outnumbers attacks — consider class
  weights, focal loss, or oversampling minority attack classes.
- **Streaming inference:** wrap `BertThreatClassifier.predict_proba()` behind a lightweight
  FastAPI service for real-time SIEM integration instead of batch CSV upload.
- **Monitoring:** track prediction drift and false-positive rate over time; retrain
  periodically as attacker techniques evolve.

## 6. Disclaimer

This project uses **synthetically generated** training data for demonstration purposes. Model
performance on the bundled dataset is not representative of real-world network traffic, which
is noisier and far more imbalanced. Validate thoroughly on real, representative data before any
production security use.
