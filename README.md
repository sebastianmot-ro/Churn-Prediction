# Customer Churn Prediction (Machine Learning)

<img width="640" height="480" alt="churn_distribution" src="https://github.com/user-attachments/assets/a1bf121a-3839-4ae8-90df-89eed3378bd0" />


## Overview
This project implements an end-to-end customer churn prediction system using supervised machine learning. The objective is to predict whether a customer is likely to churn (True) or stay (False) based on their usage behavior, service plan, and interactions with customer support. The focus is the full ML workflow, from raw data to a reusable trained model.

## Dataset
- Size: 5,000 customers
- Target: Churn (binary: True / False)
- Features: usage metrics, service plans, and account-level information

The dataset is well balanced (roughly 50% churners and 50% non-churners), which allows accuracy, F1-score, and ROC-AUC to be meaningful evaluation metrics without applying resampling techniques.

## Preprocessing & Feature Engineering
- Identifier-like columns (e.g. phone numbers, IDs) were removed to prevent data leakage.
- Preprocessing was fully automated using scikit-learn Pipelines, ensuring a clean separation between training and test data.

## Models Trained
Two models were trained and compared:
- Logistic Regression – simple, interpretable baseline
- Random Forest Classifier – non-linear model with higher capacity

Model selection was done using 5-fold Stratified Cross-Validation, optimized for ROC-AUC.

## Cross-Validation Results (Training Set)
| Model | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | ~0.87 | ~0.87 | ~0.94 |
| Random Forest | ~0.92 | ~0.92 | ~0.97 |

The Random Forest clearly outperformed the baseline and was selected as the final model.

## Final Test Set Evaluation
Confusion Matrix:

<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1f2c190a-81ac-4cbf-bbd6-f73ed97985d7" />

- True Positives: 474
- True Negatives: 463

Both false positives and false negatives are kept very low.

Overall test accuracy: 93.7%

## Feature Importance (Random Forest)
<img width="640" height="480" alt="feature_importance_top15" src="https://github.com/user-attachments/assets/94a8a5a7-2a29-4d4f-8c51-0a7608b3b5df" />

The most influential features include:
- Night Charge
- Day Minutes
- Night Calls
- Evening Minutes
- Customer Service Calls

These results align well with business intuition: customers with heavy usage patterns or frequent support interactions are more likely to churn.

## Precision–Recall & ROC Performance
<img width="640" height="480" alt="pr_curve" src="https://github.com/user-attachments/assets/55f04460-1f4d-4baf-9026-6fbf29755332" />
<img width="640" height="480" alt="roc_curve" src="https://github.com/user-attachments/assets/ec581722-c649-448f-a12b-2b54df940990" />


- Average Precision (AP): ~0.98
- Test ROC-AUC: ~0.98

High precision is maintained across most recall values, indicating strong separation between churners and non-churners across classification thresholds.

## Model Persistence
The final preprocessing + model pipeline is saved using joblib:
- models/churn_model.joblib

The model can be reloaded and used directly for inference without retraining.
