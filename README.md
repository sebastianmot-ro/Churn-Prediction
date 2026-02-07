# Customer Churn Prediction (Machine Learning)

## Overview

This project implements an end-to-end customer churn prediction system using supervised machine learning.
The objective is to predict whether a customer is likely to churn (True) or stay (False) based on their usage behavior, service plan, and interactions with customer support.

The focus of the project is the full ML workflow, from raw data to a reusable trained model.

## Dataset

Size: 5,000 customers

Target: Churn (binary: True / False)

Features: usage metrics, service plans, and account-level information

The dataset is well balanced (roughly 50% churners and 50% non-churners), which allows accuracy, F1-score, and ROC-AUC to be meaningful evaluation metrics without applying resampling techniques.

## Preprocessing & Feature Engineering

To prevent data leakage, identifier-like columns (e.g. phone numbers, IDs) were removed.

Preprocessing was fully automated using scikit-learn Pipelines, ensuring a clean separation between training and test data.

## Models Trained

Two models were trained and compared:

*Logistic Regression – simple, interpretable baseline*

*Random Forest Classifier – non-linear model with higher capacity*

Model selection was done using 5-fold Stratified Cross-Validation, optimized for ROC-AUC.

## Cross-Validation Results (Training Set)

| Model  | Accuracy | F1 | ROC-AUC |
| -------| --------- | ---- | ---- |
| Logistic Regression  | ~0.87 |	~0.87| ~0.94|
| Random Forest  | ~0.92  | ~0.92 | ~0.97|

*The Random Forest clearly outperformed the baseline and was selected as the final model.*

## Final Test Set Evaluation
Confusion Matrix

True Positives: 474

True Negatives: 463

Both false positives and false negatives are kept very low.

Overall test accuracy: 93.7%

## Feature Importance (Random Forest)

The most influential features include:

Night Charge

Day Minutes

Night Calls

Evening Minutes

Customer Service Calls

These results align well with business intuition: customers with heavy usage patterns or frequent support interactions are more likely to churn.

## Precision–Recall & ROC Performance
Average Precision (AP): ~0.98

High precision is maintained across most recall values

Test ROC-AUC: ~0.98

This indicates strong separation between churners and non-churners across classification thresholds.

## Model Persistence

The final preprocessing + model pipeline is saved using joblib:

models/churn_model.joblib

The model can be reloaded and used directly for inference without retraining.
