# Fraud Detection & Transaction Monitoring System

## Project Overview
End-to-end fraud detection pipeline built on the IEEE-CIS 
credit card fraud dataset (284,807 transactions, 492 fraud 
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
1.
