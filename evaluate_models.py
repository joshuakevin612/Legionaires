"""
evaluate_models.py
===================
Computes a full metrics report (AUC, sensitivity/recall, specificity,
precision, F1, confusion matrix) for the tabular and CT models,
using the SAME train/test split logic as the training scripts so
these numbers reflect genuine held-out performance, not numbers
already seen during training.

Why not just "accuracy"? Your stroke dataset is ~95% negative class --
a model that always predicts "no stroke" scores 95% accuracy while
being clinically useless. Sensitivity (catching real positives) matters
far more here than raw accuracy, since a missed stroke risk is much
costlier than a false alarm.

Usage
-----
    python evaluate_models.py --tabular-csv stroke_prediction.csv
    python evaluate_models.py --ct-image-dir data/head_ct --ct-labels data/labels.csv \
        --ct-filename-col id --ct-label-col " hemorrhage" --ct-build-filename-from-id
    # or both in one run
    python evaluate_models.py --tabular-csv stroke_prediction.csv \
        --ct-image-dir data/head_ct --ct-labels data/labels.csv \
        --ct-filename-col id --ct-label-col " hemorrhage" --ct-build-filename-from-id

Requires: pip install scikit-learn pandas numpy joblib
(add torch + torchvision + pillow if evaluating the CT model)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Reuse the exact preprocessing used at training time, so the split and
# encoding match what the saved model was actually trained/tested on.
from train_tabular_model import FEATURE_COLUMNS, load_and_preprocess


def print_report(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # a.k.a. recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_true

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"n = {len(y_true)} | positive rate = {y_true.mean():.3%}")
    print(f"AUC-ROC:       {auc:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}  (recall -- catches real positives)")
    print(f"Specificity:   {specificity:.4f}  (correctly clears real negatives)")
    print(f"Precision:     {precision:.4f}")
    print(f"F1 score:      {f1:.4f}")
    print("\nConfusion matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0        {tn:6d}        {fp:6d}")
    print(f"  Actual 1        {fn:6d}        {tp:6d}")


def evaluate_tabular(csv_path: str):
    import joblib

    X, y = load_and_preprocess(csv_path)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model_path_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabular_model.json")
    model_path_pkl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabular_model.pkl")

    if os.path.isfile(model_path_json):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path_json)
    elif os.path.isfile(model_path_pkl):
        model = joblib.load(model_path_pkl)
    else:
        sys.exit("ERROR: no trained tabular model found (tabular_model.json/.pkl)")

    y_prob = model.predict_proba(X_test)[:, 1]
    print_report("TABULAR MODEL (held-out test set)", y_test, y_prob)


def evaluate_ct(image_dir, labels_path, filename_col, label_col, build_filename_from_id, pad_width, extension):
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from PIL import Image

    from train_ct_model import load_labels

    filenames, labels = load_labels(
        labels_path, filename_col, label_col, build_filename_from_id, pad_width, extension
    )
    keep = np.array([os.path.isfile(os.path.join(image_dir, f)) for f in filenames])
    filenames, labels = filenames[keep], labels[keep]

    _, X_test, _, y_test = train_test_split(
        filenames, labels, test_size=0.2, random_state=42, stratify=labels
    )

    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct_resnet.pt")
    if not os.path.isfile(weights_path):
        sys.exit("ERROR: ct_resnet.pt not found -- train the CT model first")

    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((224, 224)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    probs = []
    with torch.no_grad():
        for fname in X_test:
            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            prob = torch.sigmoid(model(tensor)).item()
            probs.append(prob)

    print_report("CT MODEL (held-out test set)", np.array(y_test, dtype=int), np.array(probs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabular-csv", help="Path to labeled stroke CSV to evaluate the tabular model on")
    parser.add_argument("--ct-image-dir", help="Path to CT image folder")
    parser.add_argument("--ct-labels", help="Path to CT labels CSV/Excel")
    parser.add_argument("--ct-filename-col", help="Column holding filenames or ids")
    parser.add_argument("--ct-label-col", help="Column holding the binary label")
    parser.add_argument("--ct-build-filename-from-id", action="store_true")
    parser.add_argument("--ct-pad-width", type=int, default=3)
    parser.add_argument("--ct-extension", default=".png")
    args = parser.parse_args()

    if not args.tabular_csv and not args.ct_image_dir:
        sys.exit("ERROR: provide --tabular-csv and/or --ct-image-dir (+ related CT args)")

    if args.tabular_csv:
        evaluate_tabular(args.tabular_csv)

    if args.ct_image_dir:
        evaluate_ct(
            args.ct_image_dir,
            args.ct_labels,
            args.ct_filename_col,
            args.ct_label_col,
            args.ct_build_filename_from_id,
            args.ct_pad_width,
            args.ct_extension,
        )


if __name__ == "__main__":
    main()
