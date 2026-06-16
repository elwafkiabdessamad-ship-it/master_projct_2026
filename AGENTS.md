# AGENTS.md — QSAR Aquatic Toxicity Prediction

## Project

QSAR models predicting acute aquatic toxicity (log_LC50) from the EnviroTox database. Originally regression (plateaued at R²~0.50), pivoted to binary classification (toxic at log_LC50 < 1.0). Three parallel workflows: classical ML benchmark, GNN comparison (GATv2/PNA/GIN), Transformer comparison (ChemBERTa variants).

## Setup

- Python 3.12, uv package manager
- Activate: `source .venv/bin/activate`
- pyproject.toml has **no dependencies** declared — packages installed ad-hoc via `uv pip install <pkg>`
- uv.lock is minimal (project name only, no dep tree)
- Actual key packages: RDKit 2026.3.3, torch 2.12.0, torch-geometric 2.8.0, transformers 5.10.2, scikit-learn 1.9.0, xgboost 3.2.0, lightgbm 4.6.0, pandas 3.0.3, numpy 2.4.6, matplotlib 3.10.9, seaborn 0.13.2

## Entrypoints

Notebooks are the real code — `main.py` is a stub. Execute in order:

1. `data_curation_rdkit.ipynb` — curates raw KNIME export → curated CSV
2. `qsar_classification_models_Imane.ipynb` or `qsar_classification_models_Abdou.ipynb` — classical ML benchmark
3. `qsar_deep_gnn_comparison.ipynb` — GNN benchmark vs best classical models
4. `qsar_transformer_comparison.ipynb` — ChemBERTa benchmark vs best classical models

Scripts in `scripts/` for hyperparameter tuning (run standalone: `python scripts/<name>.py`).

Deployment apps:
- `app_utils.py` — shared RDKit feature engineering (Morgan FP + 217 descriptors + VarianceThreshold)
- `app_imane.py` — Streamlit app for Imane's best model (LightGBM, AUC 0.832)
- `app_abdou.py` — Gradio app for Abdou's best model (ExtraTrees, AUC 0.895)
- Run: `streamlit run app_imane.py` / `python app_abdou.py`

## Conventions

- **No tests, no lint, no typecheck, no CI** — none configured
- `.numpy()` fails on MPS tensors; use `.cpu().numpy()` instead
- `GlobalAttention` is deprecated → use `AttentionalAggregation`
- data/*.csv files **are tracked** (`data/*.csv` commented out in .gitignore)
- models/*.pkl and models/*.pth are gitignored
- Dual-contributor notebooks exist with `_Imane` and `_Abdou` suffixes
