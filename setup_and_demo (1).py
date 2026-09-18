"""
setup_and_demo.py
------------------
Bootstraps a local demo environment for the Cross-Modal "Silent Stroke"
Early Warning Engine Streamlit app.

What it does:
  1. Creates a ./demo_data folder (if it doesn't already exist).
  2. Generates demo_data/sample_ecg.csv
     - 1000 float values simulating a single-lead ECG waveform
       (baseline sinus rhythm + P-QRS-T complexes + light noise).
  3. Generates demo_data/sample_ct.png
     - A 224x224 grayscale synthetic image simulating a medical CT
       slice (radial skull-like intensity gradient + simulated
       ventricles + speckle noise), built with PIL/NumPy.

Run:
    python setup_and_demo.py
"""

import os
import csv
import numpy as np
from PIL import Image

DEMO_DIR = "demo_data"
ECG_PATH = os.path.join(DEMO_DIR, "sample_ecg.csv")
CT_PATH = os.path.join(DEMO_DIR, "sample_ct.png")

N_ECG_SAMPLES = 1000
CT_SIZE = 224

RNG_SEED = 42


def ensure_demo_dir() -> None:
    os.makedirs(DEMO_DIR, exist_ok=True)
    print(f"[OK] Directory ready: ./{DEMO_DIR}/")


def generate_synthetic_ecg(n_samples: int = N_ECG_SAMPLES, seed: int = RNG_SEED) -> np.ndarray:
    """
    Builds a simple synthetic single-lead ECG-like waveform:
    a repeating P-QRS-T pattern on a slow baseline wander, plus
    light Gaussian noise. This is NOT real ECG data -- it's a
    plausible-looking stand-in for wiring up and testing the
    Module 1 (1D-CNN) ingestion path.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, n_samples)  # 10 "seconds" of signal

    # Baseline wander (slow drift)
    baseline = 0.05 * np.sin(2 * np.pi * 0.3 * t)

    # Heartbeat template built from a few Gaussian bumps:
    # P wave, Q dip, R spike, S dip, T wave
    def beat(center, amp_p=0.1, amp_qrs=1.0, amp_t=0.25, width=0.03):
        p = amp_p * np.exp(-((t - (center - 0.20)) ** 2) / (2 * (width * 2) ** 2))
        q = -0.15 * amp_qrs * np.exp(-((t - (center - 0.04)) ** 2) / (2 * (width * 0.5) ** 2))
        r = amp_qrs * np.exp(-((t - center) ** 2) / (2 * (width * 0.4) ** 2))
        s = -0.25 * amp_qrs * np.exp(-((t - (center + 0.04)) ** 2) / (2 * (width * 0.5) ** 2))
        t_wave = amp_t * np.exp(-((t - (center + 0.30)) ** 2) / (2 * (width * 3) ** 2))
        return p + q + r + s + t_wave

    signal = np.zeros_like(t)
    heart_rate_hz = 1.1  # ~66 bpm
    beat_centers = np.arange(0.5, 10, 1.0 / heart_rate_hz)
    for c in beat_centers:
        signal += beat(c)

    noise = rng.normal(0, 0.02, size=n_samples)
    ecg = baseline + signal + noise
    return ecg.astype(np.float32)


def write_ecg_csv(values: np.ndarray, path: str = ECG_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "amplitude_mv"])
        for i, v in enumerate(values):
            writer.writerow([i, f"{v:.6f}"])
    print(f"[OK] Wrote {len(values)} ECG samples -> {path}")


def generate_synthetic_ct(size: int = CT_SIZE, seed: int = RNG_SEED) -> np.ndarray:
    """
    Builds a 224x224 grayscale image that loosely resembles an axial
    CT slice: a circular skull-like boundary, softer brain tissue
    interior, two darker ventricle-like blobs, and speckle noise.
    This is a SYNTHETIC placeholder image only -- not real patient
    imaging -- intended purely to exercise the Module 3 (CNN) path.
    """
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:size, :size]
    cx, cy = size / 2, size / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = size / 2

    img = np.zeros((size, size), dtype=np.float32)

    # Background (outside skull) stays near-black
    outside = r > max_r * 0.92
    # Skull ring (bright, dense bone)
    skull_ring = (r <= max_r * 0.92) & (r > max_r * 0.80)
    # Brain tissue (mid-gray, softly graded)
    brain = r <= max_r * 0.80

    img[outside] = 5
    img[skull_ring] = 220
    # Radial soft gradient for brain tissue
    tissue_gradient = 90 + 40 * (1 - (r / (max_r * 0.80)))
    img[brain] = tissue_gradient[brain]

    # Simulated ventricles: two darker elliptical blobs near center
    for dx, dy in [(-18, 0), (18, 0)]:
        vx, vy = cx + dx, cy + dy
        ellipse = ((x - vx) ** 2 / (22 ** 2) + (y - vy) ** 2 / (12 ** 2)) <= 1
        img[ellipse & brain] = 30

    # Speckle / acquisition noise
    noise = rng.normal(0, 6, size=(size, size))
    img = img + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def write_ct_png(image_array: np.ndarray, path: str = CT_PATH) -> None:
    Image.fromarray(image_array, mode="L").save(path)
    print(f"[OK] Wrote {image_array.shape[0]}x{image_array.shape[1]} grayscale CT slice -> {path}")


def main() -> None:
    print("Setting up demo data for the Silent Stroke Early Warning Engine...\n")
    ensure_demo_dir()

    ecg_values = generate_synthetic_ecg()
    write_ecg_csv(ecg_values)

    ct_image = generate_synthetic_ct()
    write_ct_png(ct_image)

    print("\nDone. Demo assets are ready in ./demo_data/")
    print(f"  - {ECG_PATH}")
    print(f"  - {CT_PATH}")
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
