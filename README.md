# Fraud Detection & Transaction Monitoring System

## Project Overview
End-to-end fraud detection pipeline built on credit card fraud dataset (284,807 transactions, 492 fraud 
cases — 0.17% fraud rate). Combines machine learning with 
domain expertise from 7+ years of fraud investigation at 
PayPal to build a production-grade fraud scoring system.

## Business Objective
- Detect fraudulent transactions while minimising operational cost
- Explain model decisions using SHAP for fraud analyst trust
- Optimise decision threshold using both statistical and 
  business cost frameworks
- Deploy fraud scoring model as interactive Streamlit application

## Project Structure
| File | Description |
|------|-------------|
| 01_Fraud_Detection_Models_and_Threshold_Optimisation.ipynb | End-to-end fraud detection pipeline |
| app.py | Streamlit web application for fraud scoring |
| eval.py | Model evaluation utilities |

## Technical Approach

### Models Trained
Three classifiers benchmarked with hyperparameter tuning 
optimised for severe class imbalance:
- Logistic Regression with balanced and custom class weights
- Random Forest with class weight tuning
- XGBoost with scale_pos_weight optimisation

### Evaluation Framework
Primary metric: AUPRC — more informative than ROC-AUC 
for severely imbalanced fraud datasets

Full metrics suite:
- Precision, Recall, F1-Score
- ROC-AUC, AUPRC, G-Mean, MCC
- Yellowbrick visualisations — ClassificationReport, 
  ROCAUC, PrecisionRecallCurve, DiscriminationThreshold

### Model Explainability — SHAP
Applied SHAP (SHapley Additive Explanations) to explain 
individual fraud predictions — identifying key fraud signals 
and validating model logic against real-world fraud typologies.

Critical for regulatory compliance and fraud analyst 
trust in production environments.

### Business Cost Analysis
Defined cost matrix reflecting operational reality:
- False Negative (missed fraud): Cost = 100
- False Positive (false alarm): Cost = 10
- True Negative / True Positive: Cost = 1

Demonstrated the Accuracy Paradox — a naive model predicting 
all transactions as legitimate achieves 99.8% accuracy 
but catches zero fraud.

### Dual Threshold Optimisation
Systematic threshold tuning across 100 values (0 to 1):
1. AUPRC optimisation — maximising fraud class performance
2. Total Cost optimisation — minimising operational loss

Results reveal the tension between statistical performance 
and business cost minimisation — the final threshold choice 
depends on the bank's risk appetite.

### Streamlit Deployment
Final model deployed as an interactive web application 
enabling real-time fraud probability scoring on new 
transaction data.

## Domain Context
Feature engineering and fraud typology selection grounded 
in 7+ years of hands-on fraud investigation at PayPal:
- Velocity abuse patterns
- Behavioural anomalies
- Account takeover signals
- Mule account detection typologies

This domain expertise bridges the gap between ML modelling 
and real-world fraud operations — ensuring the model 
captures meaningful fraud signals rather than statistical noise.

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Imbalanced-learn
- SHAP
- Yellowbrick
- Matplotlib, Seaborn
- Streamlit

## Dataset
Credit Card Fraud Detection Dataset
- 284,807 transactions
- 492 fraud cases (0.17% fraud rate)
- 28 PCA-transformed features (V1-V28)
- Download from Kaggle:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Place as: data/creditcard.csv

## Key Results
- XGBoost achieves lowest total business cost across all models
- AUPRC-optimised threshold differs significantly from default 0.5 — confirming threshold tuning is essential
- SHAP confirms model decisions align with known fraud signal typologies from PayPal domain knowledge
- Cost-based threshold reveals precision-recall tradeoff that purely statistical metrics miss

## Live App
👉 **[Click here to open the live fraud scoring app](https://manzoor-fraud-detection.streamlit.app)**

## Author
**Manzoor Syiemlieh**
Data Scientist | Fraud & Risk Analytics | 7+ Years Fintech @ PayPal
[LinkedIn](https://www.linkedin.com/in/manzoor-syiemlieh)
[GitHub](https://github.com/manzoor-syiemlieh)
