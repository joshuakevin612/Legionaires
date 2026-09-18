"""
train_ecg_model.py
===================
Trains the 1D CNN used by `predict_ecg_risk()` in ml_engine.py on a real
labeled ECG dataset, and saves the weights so ml_engine.py picks them up
automatically.

You need an ECG dataset for this -- it wasn't part of what you uploaded.
Good free options if you don't have one yet:
  - PhysioNet MIT-BIH Arrhythmia Database
  - PTB-XL (physionet.org/content/ptb-xl)
  - Kaggle "ECG Heartbeat Categorization Dataset" (already resampled to
    fixed-length beats, closest match to this contract's length-1000 format)

--------------------------------------------------------------------------
EXPECTED DATA LAYOUT
--------------------------------------------------------------------------
This script expects ONE of:

(a) A single CSV where each row is one ECG sample:
        col_0, col_1, ..., col_999, label
    i.e. 1000 signal columns + a trailing 0/1 label column.

(b) A folder of individual .npy files (one array of length 1000 each),
    plus a labels.csv with two columns: filename,label

Adjust `load_dataset()` below to match whatever shape your specific
dataset actually comes in -- ECG datasets vary a lot in format, this is
the one part of the pipeline you'll most likely need to hand-adapt.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python train_ecg_model.py --csv /path/to/ecg_dataset.csv
    # or
    python train_ecg_model.py --npy-dir /path/to/npy_folder --labels /path/to/labels.csv

Output:
    ecg_cnn.pt   -- PyTorch state_dict, auto-loaded by ml_engine.py's
                    predict_ecg_risk() the next time it runs.

Requires: pip install torch pandas numpy scikit-learn
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.exit("ERROR: this script requires PyTorch. Install with: pip install torch")

from sklearn.model_selection import train_test_split

_ECG_LENGTH = 1000


# --------------------------------------------------------------------------
# Must exactly match the _ECGCNN class in ml_engine.py so the saved
# state_dict loads back in correctly.
# --------------------------------------------------------------------------
class ECGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        flattened_size = 32 * (_ECG_LENGTH // 4)
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


def load_dataset_from_csv(csv_path: str):
    """Layout (a): 1000 signal columns + trailing label column."""
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :_ECG_LENGTH].to_numpy(dtype=np.float32)
    y = df.iloc[:, _ECG_LENGTH].to_numpy(dtype=np.float32)
    return X, y


def load_dataset_from_npy_dir(npy_dir: str, labels_csv: str):
    """Layout (b): folder of .npy files + a filename,label CSV."""
    labels_df = pd.read_csv(labels_csv)
    signals, labels = [], []
    for _, row in labels_df.iterrows():
        arr = np.load(os.path.join(npy_dir, row["filename"])).astype(np.float32).flatten()
        if arr.size != _ECG_LENGTH:
            if arr.size > _ECG_LENGTH:
                arr = arr[:_ECG_LENGTH]
            else:
                arr = np.pad(arr, (0, _ECG_LENGTH - arr.size), mode="edge")
        signals.append(arr)
        labels.append(float(row["label"]))
    return np.stack(signals), np.array(labels, dtype=np.float32)


def normalize_per_sample(X: np.ndarray) -> np.ndarray:
    """Z-score each signal individually -- matches the normalization
    predict_ecg_risk() applies at inference time."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mean) / std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV: 1000 signal cols + label col")
    parser.add_argument("--npy-dir", help="Folder of per-sample .npy signal files")
    parser.add_argument("--labels", help="labels.csv (filename,label) -- required with --npy-dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write ecg_cnn.pt into (default: this script's folder)",
    )
    args = parser.parse_args()

    if args.csv:
        X, y = load_dataset_from_csv(args.csv)
    elif args.npy_dir and args.labels:
        X, y = load_dataset_from_npy_dir(args.npy_dir, args.labels)
    else:
        sys.exit("ERROR: provide either --csv, or both --npy-dir and --labels")

    print(f"Loaded {len(y)} ECG samples | positive rate: {y.mean():.3%}")
    X = normalize_per_sample(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    train_ds = TensorDataset(
        torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train).unsqueeze(1)
    )
    val_ds = TensorDataset(torch.tensor(X_val).unsqueeze(1), torch.tensor(y_val).unsqueeze(1))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = ECGCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                correct += ((preds >= 0.5).float() == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    out_path = os.path.join(args.out_dir, "ecg_cnn.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved trained ECG CNN weights -> {out_path}")
    print("ml_engine.py will now load these automatically on next run.")


if __name__ == "__main__":
    main()
