# AGENTS.md — QSAR / Ecotoxicity Prediction

## Overview
QSAR models predicting acute toxicity (log_LC50) from the EnviroTox database. Data curation, classical ML, and deep learning baselines are complete. XGBoost remains the best model.

## Environment
- Python 3.12 (`.python-version`), conda venv at `master2026/`
- Activate: `conda activate ./master2026`
- `pyproject.toml` has **no dependencies** — all packages installed directly in the conda env
- `main.py` is a stub; **notebooks are the real entrypoints**
- No tests, no lint/typecheck config, no CI

## Key dependencies (pre-installed in venv)
- **RDKit** 2025.09.3 — descriptors, fingerprints, Mol2Vec
- **scikit-learn, xgboost, lightgbm, imbalanced-learn** — classical ML
- **torch, torch-geometric** — GNNs / deep learning (used in DL notebooks)
- **pandas, numpy, openpyxl, matplotlib, seaborn**
- **langchain, crewai, openai** — available, not yet used

## .gitignore (NOT empty)
Ignores: `master2026/`, `data/*.csv`, `data/*.xlsx`, `models/*.pkl`, `models/*.pth`, `figures/`, `Articles/`, `Old_notebooks/`, `*.docx`, `.ipynb_checkpoints/`.
**Do not commit** curated data, models, or figures without updating `.gitignore`.

## Data pipeline
1. **Source**: `the_final_data_base_in_KNIME.csv` (KNIME export of EnviroTox)
   - Download mirror: `https://toxric.bioinforai.tech/download`
2. **Curation**: `data_curation_rdkit.ipynb` → `data/curated_data_rdkit.csv` (2303 rows)
   - SMILES parsing, salt removal, charge neutralization, normalization + tautomer canonicalization, inorganic/metal filtering
3. **Legacy files to ignore**:
   - `Old_notebooks/` — superseded curation notebooks (gitignored)
   - `data/chemicals_Imane.csv`, `data/dataset_final_imane.csv` — earlier curation versions
   - `dataset_final.csv`, `data.txt` — orphaned legacy files

## Notebook execution order
1. `data_curation_rdkit.ipynb` — produces `data/curated_data_rdkit.csv`
2. `qsar_modeling.ipynb` — trains 11 classical regressors → `models/*.pkl`
3. `qsar_analysis.ipynb` — visual analysis → `figures/`
4. `deep_learning_models_executed.ipynb` — trains LSTM, GCN, CGCNN → `models/*.pth`
   - `deep_learning_models.ipynb` is the unexecuted template; use the `_executed` version

## Models summary

### Classical ML (qsar_modeling.ipynb)
- Features: 513 Morgan fingerprints + RDKit descriptors (from 2167 after variance/correlation filtering)
- Split: 80/20 train/test, random_state=42
- Target: log_LC50 (range [-5.00, 4.76])

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| **XGBoost** | **0.51** | **0.92** | **0.64** |
| LightGBM | 0.51 | 0.92 | 0.65 |
| RandomForest | 0.47 | 0.95 | 0.67 |
| GradientBoosting | — | — | — |
| Ridge / Lasso / ElasticNet | — | — | — |
| SVR / KNN / PLS | — | — | — |
| ExtraTrees / Bagging / AdaBoost | — | — | — |

### Deep Learning (deep_learning_models_executed.ipynb)
- Run date: 2026-05-17, device: CPU
- DL models **underperform** classical ML

| Model | Test R² | RMSE | MAE |
|-------|---------|------|-----|
| CGCNN | 0.37 | 1.04 | 0.77 |
| LSTM | 0.31 | 1.09 | 0.81 |
| GCN | 0.31 | 1.09 | 0.84 |

### Saved artifacts
- `models/*.pkl` — classical models + scalers (linear/SVR/KNN/PLS)
- `models/*.pth` — DL model state dicts (LSTM, GCN, CGCNN)
- `models/LSTM_scaler.pkl` — scaler for LSTM fingerprint input
- `models/XGBoost_feature_importance.csv`, `models/modeling_summary.txt`

## Directory structure
- `data/` — raw and curated datasets
- `models/` — trained model files (pkl + pth)
- `figures/` — generated plots (subdir `abdek_figures/`)
- `Articles/` — reference papers (gitignored)
- `Old_notebooks/` — legacy curation notebooks (gitignored)
- `Sujet_*.docx` — project briefs / thesis topics (French, gitignored)

## Likely next steps
1. Improve model performance (hyperparameter tuning, ensemble stacking, applicability domain)
2. External validation with held-out compounds or cross-species generalization
3. GNN architecture improvements (current DL models underperform classical ML)
4. Add tests, linting, type checking, and CI
