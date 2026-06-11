# AGENTS.md — QSAR / Ecotoxicity Prediction

## Overview
QSAR models predicting acute aquatic toxicity — **pivoted to binary classification (toxic/non-toxic at log_LC50 > 1)** after regression plateaued at R²≈0.50. Three parallel workflows:

1. **Classical ML benchmark** (11 models → best 3 by AUC-ROC)
2. **GNN comparison** (GATv2, PNA, GIN) vs best 3 classical
3. **Transformer comparison** (ChemBERTa variants) vs best 3 classical

## Environment
- Python 3.12 (`.python-version`)
- **Venv**: `.venv/` (uv-managed, home = `/opt/anaconda3/bin`)
- **Activate**: `source .venv/bin/activate`
- `pyproject.toml` has **no dependencies** — packages installed directly via `uv pip install`
- **Packages actually installed**: RDKit 2026.3.3, torch 2.12.0, torch-geometric 2.8.0, pandas 3.0.3, numpy 2.4.6, matplotlib 3.10.9, seaborn 0.13.2, xgboost==3.2.0, scikit-learn==1.9.0, lightgbm==4.6.0, transformers==5.10.2
- `main.py` is a stub; **notebooks are the real entrypoints**
- No tests, no lint/typecheck config, no CI

## .gitignore notes
- `data/*.csv` is **commented out** (line 5) — CSV files in `data/` ARE tracked by git
- Ignores: `master2026/`, `data/*.xlsx`, `models/*.pkl`, `models/*.pth`, `figures/`, `Articles/`, `Old_notebooks/`, `*.docx`, `.ipynb_checkpoints/`
- `.venv/` has its own nested `.gitignore` (`*`) so it's effectively ignored
- Do **not** commit large curated data, models, or figures without updating `.gitignore`

## Data pipeline
1. **Source**: `data/the_final_data_base_in_KNIME.csv` (KNIME export of EnviroTox, 2375 rows)
2. **Curation**: `data_curation_rdkit.ipynb` → produces `data/curated_data_rdkit*.csv` files
3. **Active dataset**: `data/curated_data_rdkit_Imane.csv` (1559 molecules, 4 cols: SMILES, CAS, log_LC50, n_measurements)

## Notebook execution order
1. `data_curation_rdkit.ipynb` — SMILES parsing, salt removal, normalization
2. **`qsar_classification_models.ipynb`** — benchmarks 11 classical classifiers on binary toxicity → saves `models/classical_benchmark.pkl` (top 3 names + results)
3. **`qsar_deep_gnn_comparison.ipynb`** — GATv2, PNA, GIN vs top 3 classical (requires step 2)
4. **`qsar_transformer_comparison.ipynb`** — ChemBERTa-77M-MLM, -MTR, druglike vs top 3 classical (requires step 2)

Run **step 2 first** — it generates the top-3 baseline that steps 3 and 4 load.

## Binary classification target
- **Threshold**: `log_LC50 > 1.0` → toxic (~45% of 1559 molecules ≈ balanced)
- **Metrics**: AUC-ROC (primary), accuracy, F1, sensitivity, specificity, precision

## Models summary

### Classical ML benchmark (qsar_classification_models.ipynb)
- **11 models**: LogisticRegression, RandomForest, XGBoost, LightGBM, SVM (RBF), GradientBoosting, ExtraTrees, AdaBoost, KNN, DecisionTree, GaussianNB
- **Features**: 2048-bit Morgan fingerprints (r=2) + 200+ RDKit descriptors → VarianceThreshold filtering
- **Output**: Sorted table + bar chart → selected top 3 by AUC-ROC

### GNN models (qsar_deep_gnn_comparison.ipynb)
3 architecturally distinct graph models via PyTorch Geometric:
- **GATv2**: 4-layer multi-head attention (8 heads), edge features → per-layer LayerNorm
- **PNA**: 5-layer principal neighbourhood aggregation (mean/std/max/min + 3 degree scalers), edge features
- **GIN**: 4-layer graph isomorphism network (GINEConv + sum aggregation), theoretically maximally expressive
- **Common**: rich atom feats (60-dim one-hot + 7 real) + 4-dim bond feats → GlobalAttention pooling → MLP head

### Transformer models (qsar_transformer_comparison.ipynb)
3 pretrained ChemBERTa (RoBERTa-encoder) variants:
- **ChemBERTa-77M-MLM**: 92.1M params, MLM pretrained on PubChem 77M
- **ChemBERTa-77M-MTR**: 92.1M params, multi-task regression pretrained (different pretraining objective)
- **ChemBERTa-druglike**: 92.1M params, curriculum-learning MLM on druglike molecules
- **Fine-tuning**: 2-phase (head-only 5 epochs → full 30 epochs, early stopping), cosine schedule, AdamW

### Regression models (superseded — kept for reference)
- XGBoost tuned: R²=0.50, RMSE=0.93, MAE=0.64 (`models/XGBoost_tuned_final.pkl`)
- LightGBM tuned: R²=0.48, RMSE=0.95, MAE=0.65 (`models/LightGBM_tuned.pkl`)
- Previous DL: CGCNN (R²=0.37), LSTM (R²=0.31), GCN (R²=0.31)

## Saved artifacts
- `models/classical_benchmark.pkl` — results table + top 3 model names
- `models/gnn_results.pkl` — GATv2/PNA/GIN results + classical comparison
- `models/transformer_results.pkl` — MLM/MTR/Druglike results + classical comparison
- `models/*_binary.pth` — trained model state dicts (GATv2, PNA, GIN, 3 ChemBERTa variants)
- `models/XGBoost_tuned_final.pkl`, `modeling_summary.txt`, `*_feature_importance.csv` — from regression phase
- `figures/classical_benchmark.png`, `gnn_vs_classical.png`, `transformer_vs_classical.png`

## Critical notes
- **MPS (Apple Silicon)**: `.numpy()` not supported on GPU tensors → use `.cpu().numpy()`; PNA may need CPU for std aggregator on MPS
- `GlobalAttention` deprecated → uses `AttentionalAggregation` but still functional
- All transformer models share the same tokenizer format (BPE SMILES tokenization)
- ChemBERTa-druglike uses a slightly different tokenizer (from Derify) — auto-loaded via `AutoTokenizer`
- Load model checkpoints from HuggingFace (~300MB each) on first run; cached in `~/.cache/huggingface/`
- GINEConv uses `edge_dim` parameter matching 4-dim bond features

## Directory structure
- `data/` — raw and curated datasets (CSVs tracked)
- `models/` — trained model artifacts (pickles gitignored)
- `figures/` — generated plots (gitignored)
- `Articles/`, `Old_notebooks/` — superseded/external content (gitignored)
- `scripts/` — helper scripts for hyperparameter tuning
