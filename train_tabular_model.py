"""
train_tabular_model.py
=======================
Trains the tabular stroke-risk model on a real labeled dataset
(e.g. the classic Kaggle "stroke-prediction" CSV) and saves it to disk
so `ml_engine.py` can load and use it directly, instead of falling
back to the synthetic-data / mock-formula baseline.

Usage
-----
    python train_tabular_model.py --csv /path/to/stroke_prediction.csv

Expected CSV columns (extra columns are ignored):
    age, hypertension, heart_disease, avg_glucose_level, bmi,
    smoking_status, stroke   <-- 'stroke' is the label (0/1)

Output
------
    tabular_model.json   (if xgboost is installed)   -- native XGBoost format
    tabular_model.pkl    (if xgboost is NOT installed) -- joblib RandomForest

`ml_engine.py` automatically looks for either file (in that priority
order) in its own directory and loads it at first call. If neither
file is present, it keeps behaving exactly as before (train-on-synthetic
-> baseline formula fallback), so nothing breaks if you skip this step.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

SMOKING_ENCODING = {
    "never smoked": 0.0,
    "unknown": 0.0,
    "formerly smoked": 0.5,
    "smokes": 1.0,
}

FEATURE_COLUMNS = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status",  # will be numerically encoded below
]


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURE_COLUMNS + ["stroke"] if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR: CSV is missing required column(s): {missing_cols}")

    # bmi sometimes contains 'N/A' strings or NaNs -> impute with median
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    # Encode smoking_status the SAME way predict_tabular_risk() does,
    # so train-time and inference-time encoding always match.
    df["smoking_status"] = (
        df["smoking_status"].astype(str).str.strip().str.lower().map(SMOKING_ENCODING).fillna(0.0)
    )

    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(df["age"].median())
    df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce").fillna(
        df["avg_glucose_level"].median()
    )

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["stroke"].astype(int).to_numpy()
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the labeled stroke CSV")
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write the trained model file into (default: this script's folder)",
    )
    args = parser.parse_args()

    X, y = load_and_preprocess(args.csv)
    print(f"Loaded {len(y)} rows | positive (stroke=1) rate: {y.mean():.3%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if _HAS_XGB:
        # Dataset is heavily imbalanced (~5% positive) -> weight positives
        # up via scale_pos_weight so the model doesn't just predict "0" always.
        n_neg, n_pos = np.bincount(y_train)
        scale_pos_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        print(f"\n[XGBoost] Test AUC:      {roc_auc_score(y_test, proba):.4f}")
        print(f"[XGBoost] Test Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds, digits=3))

        out_path = os.path.join(args.out_dir, "tabular_model.json")
        model.save_model(out_path)
        print(f"\nSaved trained XGBoost model -> {out_path}")

    elif _HAS_SKLEARN:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",  # same purpose as scale_pos_weight above
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        print(f"\n[RandomForest] Test AUC:      {roc_auc_score(y_test, proba):.4f}")
        print(f"[RandomForest] Test Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds, digits=3))
        print(
            "\nNOTE: xgboost is not installed in this environment, so a "
            "RandomForestClassifier was trained instead. Install xgboost "
            "and re-run this script to get a real XGBoost model -- "
            "ml_engine.py will automatically prefer tabular_model.json "
            "over tabular_model.pkl if both exist."
        )

        out_path = os.path.join(args.out_dir, "tabular_model.pkl")
        joblib.dump(model, out_path)
        print(f"\nSaved trained RandomForest model -> {out_path}")

    else:
        sys.exit("ERROR: neither xgboost nor scikit-learn is installed. Install one and retry.")


if __name__ == "__main__":
    main()
