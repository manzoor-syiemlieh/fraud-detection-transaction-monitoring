"""
Fraud Detection & Transaction Monitoring - Streamlit app.

Loads the trained XGBoost model and scores transactions in real time, either
from an uploaded CSV (batch) or from a built-in sample (single transaction).
The decision threshold and the cost assumptions are adjustable so an analyst
can see the operational trade-off, not just a raw probability.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ---------------------------------------------------------------- config
MODEL_PATH = "model/fraud_xgb.json"
FEATURES_PATH = "model/feature_columns.json"
SAMPLE_PATH = "data/sample_transactions.csv"

# cost-optimal threshold found in the notebook (cheapest total cost)
BEST_THRESHOLD = 0.071
# default cost assumptions: a missed fraud hurts 10x more than a false alarm
COST_FN_DEFAULT = 100   # missed fraud  (false negative)
COST_FP_DEFAULT = 10    # false alarm   (false positive)
COST_TP_DEFAULT = 1     # caught fraud  (true positive)
COST_TN_DEFAULT = 1     # correct pass  (true negative)

st.set_page_config(page_title="Fraud Detection & Transaction Monitoring",
                   layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    feature_cols = json.load(open(FEATURES_PATH))
    return model, feature_cols


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)


model, FEATURE_COLS = load_model()


# ---------------------------------------------------------------- helpers
def score_frame(df):
    """Return fraud probabilities for df, or (None, missing_columns)."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None, missing
    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
    return proba, []


def counts_and_cost(y_true, y_pred, c_tp, c_fp, c_fn, c_tn):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    total = c_tp * tp + c_fp * fp + c_fn * fn + c_tn * tn
    return tp, fp, fn, tn, total


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Decision settings")
threshold = st.sidebar.slider(
    "Decision threshold", 0.0, 1.0, BEST_THRESHOLD, 0.001,
    help="Transactions scoring at or above this are flagged as fraud. "
         "0.071 is the cost-optimal threshold from the notebook.")

st.sidebar.subheader("Cost per transaction")
c_fn = st.sidebar.number_input("Missed fraud (FN)", min_value=0, value=COST_FN_DEFAULT)
c_fp = st.sidebar.number_input("False alarm (FP)", min_value=0, value=COST_FP_DEFAULT)
c_tp = st.sidebar.number_input("Caught fraud (TP)", min_value=0, value=COST_TP_DEFAULT)
c_tn = st.sidebar.number_input("Correct pass (TN)", min_value=0, value=COST_TN_DEFAULT)


# ---------------------------------------------------------------- header
st.title("Fraud Detection & Transaction Monitoring")
st.caption(
    "XGBoost fraud scoring on credit-card transactions (28 anonymised PCA "
    "features, V1-V28). The model returns a fraud probability; the threshold "
    "and cost settings on the left turn that probability into a flag/no-flag "
    "decision and the resulting operational cost.")

mode = st.radio("Input", ["Score a sample transaction", "Upload a CSV"],
                horizontal=True)


# ---------------------------------------------------------------- sample mode
if mode == "Score a sample transaction":
    try:
        sample = load_sample()
    except FileNotFoundError:
        st.error("Sample file not found at data/sample_transactions.csv.")
        st.stop()

    st.write("Pull a random transaction from the held-out sample and score it.")
    if st.button("Score a random transaction"):
        row = sample.sample(1).reset_index(drop=True)
        proba, missing = score_frame(row)
        if proba is None:
            st.error(f"Sample is missing feature columns: {missing[:5]}")
            st.stop()

        score = float(proba[0])
        flagged = score >= threshold

        col1, col2 = st.columns(2)
        col1.metric("Fraud probability", f"{score:.4f}")
        col2.metric("Decision", "FLAGGED" if flagged else "Passed")

        if "Class" in row.columns:
            actual = int(row["Class"].iloc[0])
            st.write("Actual label:", "Fraud" if actual == 1 else "Legitimate")
            correct = (actual == 1) == flagged
            st.write("Model was", "correct." if correct else "wrong on this one.")

        st.dataframe(row, use_container_width=True)


# ---------------------------------------------------------------- upload mode
else:
    st.write("Upload a CSV containing the V1-V28 feature columns. A `Class` "
             "column (0/1), if present, is used to score accuracy and cost.")
    upload = st.file_uploader("Transactions CSV", type=["csv"])

    if upload is not None:
        df = pd.read_csv(upload)
        proba, missing = score_frame(df)
        if proba is None:
            shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            st.error(f"CSV is missing required feature columns: {shown}")
            st.stop()

        out = df.copy()
        out["fraud_probability"] = proba
        out["prediction"] = (proba >= threshold).astype(int)
        n_flagged = int(out["prediction"].sum())
        st.success(f"Scored {len(out):,} transactions - {n_flagged:,} flagged "
                   f"at threshold {threshold:.3f}.")

        if "Class" in df.columns:
            y_true = df["Class"].astype(int).values
            y_pred = out["prediction"].values
            tp, fp, fn, tn, total = counts_and_cost(
                y_true, y_pred, c_tp, c_fp, c_fn, c_tn)
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Frauds caught", f"{tp} / {tp + fn}")
            c2.metric("Recall", f"{recall:.2f}")
            c3.metric("Precision", f"{precision:.2f}")
            c4.metric("Total cost", f"{total:,}")

            # compare against the naive default-0.5 threshold
            y_pred_05 = (proba >= 0.5).astype(int)
            _, _, _, _, total_05 = counts_and_cost(
                y_true, y_pred_05, c_tp, c_fp, c_fn, c_tn)
            delta = total_05 - total
            st.caption(
                f"At the default 0.5 threshold the total cost would be "
                f"{total_05:,}. Your threshold of {threshold:.3f} "
                f"{'saves' if delta >= 0 else 'adds'} {abs(delta):,}.")

        st.subheader("Highest-risk transactions")
        st.dataframe(
            out.sort_values("fraud_probability", ascending=False).head(20),
            use_container_width=True)

        st.download_button(
            "Download scored CSV",
            out.to_csv(index=False).encode("utf-8"),
            "scored_transactions.csv", "text/csv")
