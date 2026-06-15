Fraud Detection & Transaction Monitoring System

An end-to-end credit-card fraud detection project: benchmark three models on a
severely imbalanced dataset, choose the right metric and decision threshold for a
business problem rather than a purely statistical one, explain the model with
SHAP, and ship a live app that scores new transactions in real time.

Built by a fraud investigator with 7+ years at PayPal, so the feature reasoning and
fraud typologies come from frontline operations, not just the data.

Live app: https://manzoor-fraud-detection.streamlit.app

Show Image


The problem

Card fraud is a needle-in-a-haystack problem: 492 frauds in 284,807 transactions
(0.17%). A model that flags nothing is 99.8% accurate and useless. The real
questions are which model separates fraud from non-fraud under this imbalance, what
metric should drive that choice, where to set the decision threshold when a missed
fraud costs far more than a false alarm, and whether the model's decisions can be
explained well enough for an analyst to trust them.

Data

Credit Card Fraud Detection (Kaggle, ULB)
— 284,807 transactions, 492 frauds, 28 PCA-anonymised features (V1-V28). The
model is trained on the 28 PCA features only. The full dataset isn't committed;
download it and place it at data/creditcard.csv to re-run the notebook. A small
labelled sample (data/sample_transactions.csv) is included so the live app's
sample-scoring works without the full file.

Approach


Benchmark three models — Logistic Regression, Random Forest, XGBoost — all
handling imbalance via class weighting / scale_pos_weight, not oversampling, so
the training distribution stays honest.
Judge on AUPRC, not accuracy or ROC-AUC. At a 0.17% positive rate, AUPRC
reflects performance on the class that matters.
Tune the threshold to a cost matrix. A missed fraud is set 10x more expensive
than a false alarm. Sweeping 100 thresholds shows the AUPRC-optimal point and the
cost-optimal point don't coincide — which is the whole point.
Explain with SHAP to confirm the model concentrates on a small set of
behavioural features, mapped to real typologies: velocity abuse, behavioural
anomalies, account takeover.


Model comparison

ModelROC-AUCAUPRCLogistic Regression0.97090.7092Random Forest0.96250.8460XGBoost0.97390.8670

ROC-AUC is nearly identical across all three (0.96-0.97) — which is exactly why it's
the wrong metric here. AUPRC spreads from 0.71 to 0.87 and cleanly separates the
models, so XGBoost is the pick.

Threshold: default vs cost-optimal (XGBoost)

ThresholdPrecisionRecallFrauds caughtDefault 0.50.8710.82781 / 98Cost-optimal 0.0710.7520.86785 / 98

Moving the threshold from 0.5 to 0.071 trades precision for recall — accepting more
false alarms to catch 4 more frauds — because under the 100:10 cost ratio that's the
cheapest outcome overall. This is the trade-off the live app lets you explore.

The live app

The app loads the trained XGBoost model and scores transactions in real time:


Score a sample transaction — pull a random record from the held-out sample and
see its fraud probability, the flag/no-flag decision, and whether the model was right.
Upload a CSV — score a batch of transactions (must contain V1-V28). If a
Class column is present, the app reports recall, precision, frauds caught, total
cost, and how that cost compares to the naive 0.5 threshold.


The decision threshold and the per-outcome costs are adjustable in the sidebar, so an
analyst can see the operational impact of moving the cut-off — which mirrors how fraud
teams actually tune a deployed model.

Repo structure

PathWhat it is01_Fraud_Detection_Models_and_Threshold_Optimisation.ipynbFull pipeline: EDA, three models, evaluation, SHAP, threshold sweep, model exportapp.pyStreamlit app — real-time scoring + threshold/cost analysismodel/fraud_xgb.jsonTrained XGBoost model (native format)model/feature_columns.jsonExpected feature order, used to validate inputsdata/sample_transactions.csvSmall labelled sample for the appreports/shap_summary.pngSHAP feature-importance summaryrequirements.txtPinned dependencies

Run it locally

bashgit clone https://github.com/manzoor-syiemlieh/fraud-detection-transaction-monitoring.git
cd fraud-detection-transaction-monitoring
pip install -r requirements.txt

# run the app (uses the bundled model + sample)
streamlit run app.py

# to re-run the full pipeline, download creditcard.csv from Kaggle into data/, then:
jupyter notebook 01_Fraud_Detection_Models_and_Threshold_Optimisation.ipynb

Limitations & next steps


Features are PCA-anonymised, so SHAP maps to typologies conceptually rather than to
named raw features.
No drift monitoring or scheduled retraining — the obvious production extension, out
of scope for a portfolio project.
Threshold and cost ratios are illustrative; in production they'd be set from the
bank's actual loss and review-capacity data.


Author

Manzoor Syiemlieh — Fraud & Risk Analytics, 7+ years (PayPal) -> Data Science
LinkedIn ·
GitHub

MIT Licensed.
