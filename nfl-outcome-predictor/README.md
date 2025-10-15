# NFL Outcome Predictor (2000–2024)

End‑to‑end, reproducible ML project to predict NFL game outcomes and apply the model to **Week 7 of the 2025 season**.

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Train the model on 2000–2024
python scripts/train.py

# 4) Predict Week 7 of 2025 (once 2025 weeks 1–6 have been played)
python scripts/predict_week7_2025.py
```

## What it does

- Downloads **play‑by‑play** and **games metadata** (2000–2024) via `nflreadpy`.
- Builds **team‑game features** from play‑by‑play (EPA, success rate, pass rate).
- Computes **rolling form features** per team (last 3 and 5 games) with strict cutoffs to avoid leakage.
- Trains a baseline **logistic regression** classifier to estimate **home win probability**.
- Evaluates with **AUC** and **Brier score** and saves artifacts to `models/`.
- Provides an inference script to fetch **2025 Week 7** schedule, build features through Week 6, and output predictions.

## Project structure

```
nfl-outcome-predictor/
├─ src/nfl_outcome/
│  ├─ __init__.py
│  ├─ data.py            # data acquisition helpers
│  ├─ features.py        # feature engineering + rolling windows (leakage-safe)
│  ├─ models.py          # training, evaluation, persistence
│  ├─ inference.py       # week-7 prediction pipeline
│  └─ utils.py           # mappings, common helpers
├─ scripts/
│  ├─ train.py
│  └─ predict_week7_2025.py
├─ data/
│  ├─ raw/.gitkeep
│  └─ processed/.gitkeep
├─ models/.gitkeep
├─ reports/.gitkeep
├─ notebooks/            # (optional) your scratch work
├─ config/
│  └─ settings.json
├─ tests/
│  └─ test_imports.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Tips

- To calibrate probabilities further, add **Vegas closing spreads/totals** (Kaggle or a paid feed) and use **isotonic** calibration.
- Try tree‑based models (XGBoost/LightGBM) using the same feature matrix for stronger baselines.
- For *true* out‑of‑time validation, keep 2023–2024 as a hold‑out and avoid shuffling across seasons.

## Environment

Add project‑specific env vars in `.env` (optional). An example is provided in `config/settings.json` for defaults.