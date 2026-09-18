"""
train_ecg_model.py
===================
Trains the 2D spectrogram-CNN (ResNet18) used by predict_ecg_risk() in
ml_engine.py -- converts each raw ECG waveform to a spectrogram image,
then fine-tunes an ImageNet-pretrained ResNet18 on it (same
transfer-learning approach as the CT model).

--------------------------------------------------------------------------
DATASET -- use one that's actually labeled for what this product claims
--------------------------------------------------------------------------
ECG contributes to stroke risk through ONE well-established mechanism:
detecting atrial fibrillation (AFib), a heart arrhythmia that causes
roughly 15-30% of ischemic strokes by letting clots form in the heart
and travel to the brain. For that clinical story to actually hold up,
train on a dataset labeled for AFib specifically -- not a generic
"heartbeat categorization" dataset that classifies unrelated beat types.

Recommended: the PhysioNet/CinC 2017 AF Classification Challenge dataset
("training2017"), free at:
    https://physionet.org/content/challenge-2017/1.0.0/

It ships ~8,500 single-lead ECG recordings (variable length, 300 Hz) as
.mat files, with a REFERENCE.csv labeling each as one of:
    N = Normal, A = AFib, O = Other rhythm, ~ = too noisy to classify

For THIS product's binary contract, treat A (AFib) as the positive
class and everything else as negative -- see `label_from_class()` below.
(Alternative: MIT-BIH Atrial Fibrillation Database, if you want longer
multi-hour recordings you'll need to segment yourself.)

--------------------------------------------------------------------------
EXPECTED DATA LAYOUT (after downloading training2017 and unzipping)
--------------------------------------------------------------------------
    training2017/
    |-- A00001.mat, A00002.mat, ...     <-- one .mat file per recording
    `-- REFERENCE.csv                    <-- columns: filename, label (N/A/O/~)

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python train_ecg_model.py --data-dir training2017 --reference training2017/REFERENCE.csv

Output:
    ecg_cnn.pt   -- PyTorch state_dict (ResNet18), auto-loaded by
                    ml_engine.py's predict_ecg_risk() the next time it runs.

Requires: pip install torch torchvision scipy pandas scikit-learn pillow
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit(
        "ERROR: this script requires torch + torchvision. "
        "Install with: pip install torch torchvision"
    )

try:
    from scipy.io import loadmat
    from scipy import signal as sp_signal
except ImportError:
    sys.exit("ERROR: this script requires scipy. Install with: pip install scipy")

from sklearn.model_selection import train_test_split

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: this script requires Pillow. Install with: pip install pillow")

_TARGET_LENGTH = 1000        # must match ml_engine.py's _ECG_LENGTH
_SAMPLING_RATE = 100         # must match ml_engine.py's _ECG_SAMPLING_RATE
_SPECTROGRAM_SIZE = 224      # must match ml_engine.py's _ECG_SPECTROGRAM_SIZE
_SOURCE_FS = 300             # training2017's actual recording rate


def label_from_class(raw_label: str) -> int:
    """A (AFib) -> 1, everything else (N/O/~) -> 0."""
    return 1 if str(raw_label).strip().upper() == "A" else 0


def load_and_resample_mat(path: str) -> np.ndarray:
    """
    Loads a training2017 .mat recording, takes the first _TARGET_LENGTH*3
    seconds' worth of raw samples (or pads if shorter), then resamples
    down from 300 Hz to 100 Hz so every signal ends up length 1000 --
    matching the contract's fixed-length assumption.
    """
    mat = loadmat(path)
    raw = mat["val"].flatten().astype(np.float64)

    # Resample from the source rate to our target rate/length.
    target_len_at_source_rate = int(_TARGET_LENGTH * _SOURCE_FS / _SAMPLING_RATE)
    if raw.size >= target_len_at_source_rate:
        raw = raw[:target_len_at_source_rate]
    else:
        raw = np.pad(raw, (0, target_len_at_source_rate - raw.size), mode="edge")

    resampled = sp_signal.resample(raw, _TARGET_LENGTH)
    return resampled.astype(np.float32)


def signal_to_spectrogram_image(signal: np.ndarray) -> np.ndarray:
    """Identical logic to ml_engine.ecg_signal_to_spectrogram_image, kept
    self-contained here so this script doesn't need to import ml_engine."""
    _, _, sxx = sp_signal.spectrogram(signal, fs=_SAMPLING_RATE, nperseg=64, noverlap=48)
    sxx_log = np.log1p(sxx)
    lo, hi = sxx_log.min(), sxx_log.max()
    normalized = np.zeros_like(sxx_log) if (hi - lo) < 1e-8 else (sxx_log - lo) / (hi - lo)
    img = Image.fromarray((normalized * 255).astype(np.uint8))
    img = img.resize((_SPECTROGRAM_SIZE, _SPECTROGRAM_SIZE))
    resized = np.array(img)
    return np.stack([resized, resized, resized], axis=-1)


class ECGSpectrogramDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        raw_signal = load_and_resample_mat(self.filepaths[idx])
        spec_img = signal_to_spectrogram_image(raw_signal)  # (224,224,3) uint8
        tensor = torch.tensor(spec_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        tensor = (tensor - torch.tensor(self.mean, dtype=torch.float32)) / torch.tensor(
            self.std, dtype=torch.float32
        )
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return tensor, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder of .mat recordings (training2017)")
    parser.add_argument("--reference", required=True, help="REFERENCE.csv: filename,label columns (no header)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write ecg_cnn.pt into (default: this script's folder)",
    )
    args = parser.parse_args()

    ref = pd.read_csv(args.reference, header=None, names=["filename", "label"])
    filepaths = [os.path.join(args.data_dir, f"{fn}.mat") for fn in ref["filename"]]
    labels = ref["label"].apply(label_from_class).to_numpy()

    keep = np.array([os.path.isfile(fp) for fp in filepaths])
    if not keep.all():
        print(f"WARNING: {(~keep).sum()} referenced .mat files not found, skipping them.")
    filepaths = [fp for fp, k in zip(filepaths, keep) if k]
    labels = labels[keep]

    print(f"Loaded {len(labels)} ECG recordings | AFib (positive) rate: {labels.mean():.3%}")

    X_train, X_val, y_train, y_val = train_test_split(
        filepaths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_ds = ECGSpectrogramDataset(X_train, y_train)
    val_ds = ECGSpectrogramDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    try:
        model = tv_models.resnet18(weights="DEFAULT")
    except Exception:
        model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # AFib is the minority class -- weight its loss up rather than letting
    # the model learn to just predict "normal" every time.
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n_train_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

            # Print progress every 10 batches (or every batch if the
            # epoch is small) so the terminal never looks frozen.
            if batch_idx % 10 == 0 or batch_idx == n_train_batches:
                print(
                    f"  epoch {epoch}/{args.epochs} | batch {batch_idx}/{n_train_batches} "
                    f"| running loss {train_loss / (batch_idx * args.batch_size):.4f}",
                    flush=True,
                )
        train_loss /= len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    out_path = os.path.join(args.out_dir, "ecg_cnn.pt")
    torch.save(model.to("cpu").state_dict(), out_path)
    print(f"\nSaved trained ECG spectrogram-CNN weights -> {out_path}")
    print("ml_engine.py will now load these automatically on next run.")


if __name__ == "__main__":
    main()
