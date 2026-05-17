# AGENTS.md — Project Master

## Overview
QSAR / ecotoxicity prediction project using the EnviroTox database. Data curation and baseline modeling are complete; next steps are model improvement and validation.

## Environment
- Python 3.12 (`.python-version`), conda venv at `master2026/`
- Activate: `conda activate ./master2026` (or point editor to `master2026/bin/python`)
- `pyproject.toml` has no dependencies — all packages are installed directly in the conda env
- `.gitignore` is empty — be careful not to commit large data/model files or temp outputs
- `main.py` is a stub placeholder; notebooks are the real entrypoints
- No tests, no lint/typecheck config, no CI

## Key dependencies (pre-installed in venv)
- **RDKit** `2025.09.3` — molecular descriptors, fingerprints, Mol2Vec
- **scikit-learn, xgboost, lightgbm, imbalanced-learn** — classical ML
- **torch-geometric** — GNNs (available, not yet used)
- **pandas, numpy, openpyxl** — data wrangling
- **matplotlib, seaborn** — visualization
- **langchain, crewai, openai** — LLM pipeline options (available, not yet used)

## Data pipeline
1. **Source**: `the_final_data_base_in_KNIME.csv` (KNIME export of EnviroTox)
   - Raw xlsx at `data/envirotox_20260415204209.xlsx` (older snapshot)
   - Download mirror: `https://toxric.bioinforai.tech/download`
2. **Curation**: `data_curation_rdkit.ipynb` — SMILES parsing, salt removal, charge neutralization, normalization + tautomer canonicalization, inorganic/metal filtering → `data/curated_data_rdkit.csv` (2303 rows)
3. **Legacy notebooks to ignore**: `data_curation.ipynb`, `data_curation copy.ipynb` — superseded by the RDKit version
4. **Legacy data**: `data/chemicals_Imane.csv`, `data/dataset_final_imane.csv` — earlier curation versions

## Modeling (complete)
- **Notebook**: `qsar_modeling.ipynb` — generates Morgan fingerprints + RDKit descriptors, applies variance/correlation feature filtering (513 features from 2167), trains 11 regressors, saves models + scalers to `models/`
- **Notebook**: `qsar_analysis.ipynb` — visual analysis of log_LC50 distribution and model performance comparison, outputs to `figures/`
- **Target**: `log_LC50` (range [-5.00, 4.76])
- **Best model**: XGBoost — Test R²=0.51, RMSE=0.92, MAE=0.64
- **Saved artifacts**: `models/*.pkl` (11 trained models + scalers for linear/SVR/KNN/PLS), `models/XGBoost_feature_importance.csv`, `models/modeling_summary.txt`
- Split: 80/20 train/test, random_state=42

## Notebook execution order
1. `data_curation_rdkit.ipynb` — produces `data/curated_data_rdkit.csv`
2. `qsar_modeling.ipynb` — reads curated data, trains models, saves to `models/`
3. `qsar_analysis.ipynb` — reads curated data + models, generates figures

## Directory structure
- `data/` — raw and curated datasets
- `models/` — trained model `.pkl` files and scalers
- `figures/` — generated plots (subdir `abdek_figures/`)
- `Articles/` — reference papers
- `Sujet_*.docx` — project briefs / thesis topics (French)

## Likely next steps
1. Improve model performance (hyperparameter tuning, ensemble stacking, applicability domain)
2. External validation with held-out compounds or cross-species generalization
3. GNN implementation via torch-geometric
4. Add tests, linting, type checking, and version control
