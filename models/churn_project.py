import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# Load the churn dataset (expects churn.csv one level up from this script)
df = pd.read_csv("../churn.csv")


print("Shape:", df.shape)
print(df.head())
print(df.info())

# Quick data quality check: missing values and target balance
print("\nMissing values per column (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

TARGET_COL = "Churn?"
print("\nTarget distribution:")
print(df[TARGET_COL].value_counts(dropna=False))
print("\nTarget %:")
print(df[TARGET_COL].value_counts(normalize=True, dropna=False) * 100)

# Save a simple class-balance plot for reporting
ax = df[TARGET_COL].value_counts().plot(kind="bar")
plt.title("Churn distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"))
plt.close()
print("Saved plot: plots/churn_distribution.png")

# Drop identifier-style columns so the model doesn't learn from non-generalizable IDs
for col in ["customerID", "CustomerID", "id", "ID"]:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped ID column: {col}")

df = df.drop(columns=["Phone"])


# Convert the target column into a binary label (1 = churn, 0 = not churn)
if df[TARGET_COL].dtype == "object":
    df[TARGET_COL] = df[TARGET_COL].str.strip().map({"True.": 1, "False.": 0})
    if df[TARGET_COL].isna().any():
        print("\n[Warning] Target has unmapped values. Unique target values:")
        print(df[TARGET_COL].unique())

# Attempt numeric conversion for object columns where values look numeric
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = pd.to_numeric(df[c], errors="ignore")

# Add a simple derived feature (average monthly charge), guarding against tenure=0
if "MonthlyCharges" in df.columns and "tenure" in df.columns:
    df["AvgChargePerMonth"] = df["MonthlyCharges"] / (df["tenure"].replace(0, np.nan))

# Split into features (X) and label (y) for supervised learning
X = df.drop(columns=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().map({"True.": 1, "False.": 0})

y = df[TARGET_COL].astype(int)

# Hold out a stratified test set to preserve churn class ratio in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Identify numeric vs categorical columns so we can preprocess them appropriately
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# Numeric: median impute + standardize for scale-sensitive models (e.g., logistic regression)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical: impute most frequent + one-hot encode to turn categories into model-ready columns
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# ColumnTransformer applies the right preprocessing to each column subset
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# End-to-end pipelines: preprocessing + model in one object (prevents train/test leakage)
log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

models = {
    "LogisticRegression": log_reg,
    "RandomForest": rf
}

# Helper to evaluate a trained model on the test set and generate standard plots/metrics
def evaluate_on_test(model, X_test, y_test, model_name="model"):
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} - Test Evaluation ===")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")

        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC Curve - {model_name}")
        plt.tight_layout()

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Precision-Recall - {model_name}")
        plt.tight_layout()


# Cross-validate on training data to choose a model without touching the test set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

cv_results = {}

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    cv_results[name] = {k: float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}

print("\n=== CV Mean Scores (train only) ===")
for name, res in cv_results.items():
    print(name, "->", {k.replace("test_", ""): round(v, 4) for k, v in res.items()})

# Select the best model by ROC-AUC (typical for churn); falls back via -1 if missing
best_name = max(cv_results, key=lambda n: cv_results[n].get("test_roc_auc", -1))
best_model = models[best_name]

print(f"\nBest model by CV ROC-AUC: {best_name}")

import os
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

os.makedirs("plots", exist_ok=True)

# Centralized plot saver to keep outputs consistent and avoid repeated boilerplate
def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(os.path.join("plots", filename))
    plt.close()
    print(f"Saved plot: plots/{filename}")


# Train the selected model on the full training set, then evaluate once on the held-out test set
best_model = models[best_name]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\n=== TEST SET EVALUATION ===")
print("Model:", best_name)
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title(f"Confusion Matrix - {best_name}")
save_plot("confusion_matrix.png")

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test ROC-AUC: {auc:.4f}")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"ROC Curve - {best_name}")
    save_plot("roc_curve.png")

    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Precision-Recall Curve - {best_name}")
    save_plot("pr_curve.png")

import joblib

# Persist the trained pipeline so preprocessing + model are saved together for deployment/use
os.makedirs("models", exist_ok=True)
MODEL_PATH = "models/churn_model.joblib"

joblib.dump(best_model, MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH}")

# Sanity check: reload the model and run a few predictions
loaded = joblib.load(MODEL_PATH)
print("Reload OK. Sample preds:", loaded.predict(X_test.iloc[:5]))

import numpy as np

# For RandomForest, export a quick feature-importance chart for interpretability
if best_name == "RandomForest":
    preprocessor = best_model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    rf_model = best_model.named_steps["model"]
    importances = rf_model.feature_importances_

    idx = np.argsort(importances)[::-1][:15]
    top_features = feature_names[idx]
    top_importances = importances[idx]

    plt.figure()
    plt.barh(top_features[::-1], top_importances[::-1])
    plt.title("Top 15 Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    save_plot("feature_importance_top15.png")


# Refit (same data) and run the helper evaluation to print metrics and generate plots
best_model.fit(X_train, y_train)
evaluate_on_test(best_model, X_test, y_test, model_name=best_name)

# Save again to ensure the latest fitted model is on disk
MODEL_PATH = "models/churn_model.joblib"
joblib.dump(best_model, MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH}")

loaded = joblib.load(MODEL_PATH)
print("Loaded model ok. Test prediction sample:", loaded.predict(X_test.iloc[:5]))
