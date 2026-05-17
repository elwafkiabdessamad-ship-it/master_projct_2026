# QSAR / Ecotoxicity Prediction

QSAR models for predicting acute toxicity (log_LC50) using the EnviroTox database.

## Quick Start

```bash
conda activate ./master2026
```

Run notebooks in order:
1. `data_curation_rdkit.ipynb` — curate raw data → `data/curated_data_rdkit.csv`
2. `qsar_modeling.ipynb` — train 11 regressors → `models/`
3. `qsar_analysis.ipynb` — visualize results → `figures/`

## Project Structure

| Path | Description |
|------|-------------|
| `data_curation_rdkit.ipynb` | SMILES parsing, salt removal, neutralization, normalization |
| `qsar_modeling.ipynb` | Morgan fingerprints + RDKit descriptors, 11 models |
| `qsar_analysis.ipynb` | Performance comparison, feature importance, plots |
| `data/` | Raw and curated datasets |
| `models/` | Trained `.pkl` files and scalers |
| `figures/` | Generated plots |
| `Articles/` | Reference papers |

## Data

- **Source**: EnviroTox database via KNIME export (`the_final_data_base_in_KNIME.csv`)
- **Download**: https://toxric.bioinforai.tech/download
- **Curated**: 2,303 compounds after filtering inorganics/metals and standardizing structures

## Models

| Model | Status |
|-------|--------|
| Ridge, Lasso, ElasticNet | Trained + scaler |
| SVR, KNN, PLS | Trained + scaler |
| RandomForest, ExtraTrees, Bagging | Trained |
| GradientBoosting, AdaBoost | Trained |
| XGBoost | **Best** — R²=0.51, RMSE=0.92, MAE=0.64 |
| LightGBM | Trained |

Features: 513 (from 2,167 after variance/correlation filtering). Split: 80/20, random_state=42.

## Dependencies

All pre-installed in the `master2026` conda environment:

- RDKit 2025.09.3
- scikit-learn, xgboost, lightgbm, imbalanced-learn
- torch-geometric (available, not yet used)
- pandas, numpy, matplotlib, seaborn

## Next Steps

1. Hyperparameter tuning and ensemble stacking
2. Applicability domain definition
3. External validation / cross-species generalization
4. GNN implementation (torch-geometric)
5. Tests, linting, type checking, CI
