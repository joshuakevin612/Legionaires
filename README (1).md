# Cross-Modal Silent Stroke & Cerebrovascular Early Warning Engine

A Streamlit dashboard that fuses three independent risk models — clinical
tabular data, an ECG waveform, and a brain CT scan — into one composite
stroke-risk tier (Low / Moderate / High).

## Architecture

```
Streamlit dashboard (app.py)
        │
        ├── predict_tabular_risk(dict)        → float  [0, 1]
        ├── predict_ecg_risk(file_path)       → (float, list[1000])
        ├── predict_ct_risk(file_path)        → float  [0, 1]
        │
        └── calculate_composite_risk(p_tab, p_ecg, p_ct)
                → (risk_percentage: float [0, 100], tier: str)
                Composite = 0.35·p_tab + 0.35·p_ecg + 0.30·p_ct
                Tiers: Low (<35%) / Moderate (35–69%) / High (≥70%)
```

All inference logic lives in `ml_engine.py`, which is dependency-optional:
if `torch`/`torchvision`/`xgboost` aren't installed, or a trained weights
file is missing, each function falls back to a deterministic,
clinically-informed mock so the app never hard-crashes.

## Project status

| Component | Status |
|---|---|
| Tabular model (RandomForest, Kaggle stroke-prediction CSV) | ✅ Trained — AUC 0.83, accuracy 0.89 on held-out test set |
| CT model (ResNet18, Kaggle Head-CT Hemorrhage) | 🟡 Trained — batch validation in progress |
| ECG model (1D CNN) | ⚪ Architecture + training script ready; no real dataset sourced yet — currently running on the deterministic synthetic fallback |
| Dashboard wiring + composite-score scale bug | ✅ Fixed |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Retraining the models

```bash
python train_tabular_model.py --csv path/to/stroke_prediction.csv
python train_ct_model.py --data-dir path/to/head_ct --labels path/to/labels.csv
python train_ecg_model.py --csv path/to/ecg_dataset.csv
```

Each script writes its output artifact (`tabular_model.pkl`/`.json`,
`ct_resnet.pt`, `ecg_cnn.pt`) into the project root, where `ml_engine.py`
picks it up automatically on the next run — no code changes needed.

## Datasets

Not included in this repo (see `.gitignore`) due to size and
redistribution licensing. Source them yourself:

- **Tabular**: [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **CT**: [Kaggle — Head CT - Hemorrhage](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage) (Felipe Kitamura)
- **ECG**: not yet sourced. Candidates: PhysioNet MIT-BIH, PTB-XL, or
  Kaggle "ECG Heartbeat Categorization Dataset"

## Known limitations

- ECG model has not been trained on real data yet — scores are from a
  deterministic synthetic fallback, not a learned classifier.
- CT model's batch validation accuracy has not yet been confirmed; early
  spot checks suggest possible bias toward "normal" classification.
- This is a research/educational prototype, not a validated clinical
  diagnostic tool.

## Disclaimer

For research and educational purposes only. Not intended for clinical
use or as a substitute for professional medical diagnosis.
