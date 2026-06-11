"""
XGBoost hyperparameter tuning for QSAR log_LC50 prediction.

Replicates the exact data pipeline from qsar_modeling.ipynb then
runs RandomizedSearchCV over a broad hyperparameter space.
"""

import os, warnings, pickle, time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import (
    train_test_split, KFold, RandomizedSearchCV, cross_val_predict
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
RNG = 42

# ── paths ──────────────────────────────────────────────────────────────
DATA_IN  = 'data/the_final_data_base_in_KNIME.csv'
OUT_DIR  = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

# ========================================================================
# 1. LOAD
# ========================================================================
print('── 1. Load KNIME CSV ──')
df = pd.read_csv(DATA_IN)
print(f'  Loaded: {df.shape[0]} rows, {df.shape[1]} cols')

# drop metadata rows (3 rows with "Meta" in CAS_str)
meta = df['CAS_str'].astype(str).str.contains('Meta', na=False)
df = df[~meta].reset_index(drop=True)
print(f'  After Meta drop: {df.shape[0]} rows')

# ========================================================================
# 2. CURATION  (replicates data_curation_rdkit.ipynb steps 2-7)
# ========================================================================
print('── 2. Curate ──')

organic_set = {'H','B','C','N','O','F','Si','P','S','Cl','Se','Br','I'}

def parse_smiles(s):
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def remove_salt(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1:
        return mol
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    try:
        Chem.SanitizeMol(largest)
        return largest
    except Exception:
        return mol

def is_inorganic(mol):
    """True if mol has no carbon or contains element outside organic_set."""
    has_c = any(at.GetAtomicNum() == 6 for at in mol.GetAtoms())
    if not has_c:
        return True
    for at in mol.GetAtoms():
        if at.GetSymbol() not in organic_set:
            return True
    return False

# parse
df['mol'] = df['SMILES'].apply(parse_smiles)
before = len(df)
df = df.dropna(subset=['mol']).reset_index(drop=True)
print(f'  SMILES parsed: {len(df)} / {before} valid')

# remove salts
df['mol'] = df['mol'].apply(remove_salt)

# neutralise charges
uncharger = rdMolStandardize.Uncharger()
df['mol'] = df['mol'].apply(uncharger.uncharge)

# standardise (normalise + tautomer canonicalize)
normalizer = rdMolStandardize.Normalizer()
tautomerizer = rdMolStandardize.TautomerEnumerator()

def standardize(mol):
    try:
        mol = normalizer.normalize(mol)
        mol = tautomerizer.Canonicalize(mol)
    except Exception:
        pass
    return mol

df['mol'] = df['mol'].apply(standardize)

# filter inorganics
df['inorganic'] = df['mol'].apply(is_inorganic)
n_inorg = df['inorganic'].sum()
df = df[~df['inorganic']].reset_index(drop=True)
print(f'  Inorganics removed: {n_inorg}')
print(f'  After curation: {df.shape[0]} rows, {df.shape[1]} cols')

# ========================================================================
# 3. FEATURE ENGINEERING  (replicates qsar_modeling.ipynb steps 2-5)
# ========================================================================
print('── 3. Feature engineering ──')

# preserve curated SMILES for later use
df['SMILES_curated'] = df['mol'].apply(lambda m: Chem.MolToSmiles(m))

meta_cols = ['CAS_str', 'Chemical_name', 'SMILES', 'mlecule_H_final',
             'SMILES_curated', 'mol', 'inorganic']
target_col = 'log_LC50'
desc_cols = [c for c in df.columns
             if c not in meta_cols + [target_col, 'n_measurements']]

X_desc = df[desc_cols].values
y = df[target_col].values
print(f'  Descriptors: {X_desc.shape[1]}')

# Morgan fingerprints (radius=2, 2048 bits)
mols = df['mol'].tolist()
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(m) for m in mols]
X_fp = np.array(fps)
print(f'  Fingerprints: {X_fp.shape[1]}')

X_combined = np.hstack([X_desc, X_fp])
all_feature_names = (desc_cols +
                     [f'FP_{i}' for i in range(X_fp.shape[1])])
print(f'  Combined features: {X_combined.shape[1]}')

# Variance threshold
var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_combined)
kept_mask = var_thresh.get_support()
kept_names = [all_feature_names[i] for i in range(len(all_feature_names))
              if kept_mask[i]]
print(f'  After variance filter: {X_var.shape[1]} (removed '
      f'{X_combined.shape[1] - X_var.shape[1]})')

# Correlation filter (|r| > 0.95)
corr_threshold = 0.95
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
)
to_drop = [col for col in upper_tri.columns
           if any(upper_tri[col] > corr_threshold)]
keep_idx = [i for i in range(X_var.shape[1]) if i not in to_drop]
X_final = X_var[:, keep_idx]
final_names = [kept_names[i] for i in keep_idx]
print(f'  After correlation filter: {X_final.shape[1]} (removed '
      f'{len(to_drop)})')
print(f'  Final features: {X_final.shape[1]}')

# ========================================================================
# 4. TRAIN / TEST SPLIT
# ========================================================================
print('── 4. Train/test split ──')
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=RNG
)
print(f'  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# ========================================================================
# 5. BASELINE  (default XGBoost params from notebook)
# ========================================================================
print('── 5. Baseline (notebook defaults) ──')
baseline = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    random_state=RNG, n_jobs=-1, verbosity=0
)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
base_r2  = r2_score(y_test, y_pred_base)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
base_mae = mean_absolute_error(y_test, y_pred_base)
print(f'  R²={base_r2:.4f}  RMSE={base_rmse:.4f}  MAE={base_mae:.4f}')

# ========================================================================
# 6. HYPERPARAMETER TUNING  (RandomizedSearchCV, 5-fold)
# ========================================================================
print('── 6. Randomized search ──')

param_dist = {
    'n_estimators':       [200, 400, 600, 800, 1000],
    'max_depth':          [3, 4, 5, 6, 7, 8, 10],
    'learning_rate':      [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    'subsample':          [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':   [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight':   [1, 2, 3, 5, 7],
    'gamma':              [0, 0.05, 0.1, 0.2, 0.5],
    'reg_alpha':          [0, 0.01, 0.1, 1.0, 5.0],
    'reg_lambda':         [0, 0.01, 0.1, 1.0, 5.0],
}

xgb_model = xgb.XGBRegressor(random_state=RNG, n_jobs=-1, verbosity=0)
cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
search = RandomizedSearchCV(
    xgb_model, param_dist, n_iter=100,
    scoring='r2', cv=cv, random_state=RNG, n_jobs=-1,
    verbose=1, return_train_score=True,
)

t0 = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - t0

print(f'\n  Search completed in {elapsed:.1f}s')
print(f'  Best CV R²: {search.best_score_:.4f}')
print(f'  Best params: {search.best_params_}')

# ========================================================================
# 7. EVALUATE BEST MODEL ON TEST SET
# ========================================================================
print('── 7. Test set evaluation ──')
best = search.best_estimator_
y_pred_best = best.predict(X_test)
best_r2  = r2_score(y_test, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
best_mae = mean_absolute_error(y_test, y_pred_best)
print(f'  Baseline  →  R²={base_r2:.4f}  RMSE={base_rmse:.4f}  MAE={base_mae:.4f}')
print(f'  Tuned     →  R²={best_r2:.4f}  RMSE={best_rmse:.4f}  MAE={best_mae:.4f}')
print(f'  Δ R²: {best_r2 - base_r2:+.4f}')

# train-set metrics for overfit check
y_train_pred = best.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f'  Train (tuned): R²={train_r2:.4f}  RMSE={train_rmse:.4f}')
print(f'  Gap: {train_r2 - best_r2:.4f}')

# ========================================================================
# 8. SAVE
# ========================================================================
print('── 8. Save ──')
model_path = os.path.join(OUT_DIR, 'XGBoost_tuned.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best, f)

results = {
    'baseline':  {'R2': base_r2,  'RMSE': base_rmse,  'MAE': base_mae},
    'tuned':     {'R2': best_r2,  'RMSE': best_rmse,  'MAE': best_mae,
                  'train_R2': train_r2, 'train_RMSE': train_rmse,
                  'gap': train_r2 - best_r2},
    'best_params': search.best_params_,
    'cv_scores': {
        'mean': search.best_score_,
        'std':  search.cv_results_['std_test_score'][search.best_index_],
    },
}
results_path = os.path.join(OUT_DIR, 'XGBoost_tuning_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f'  Model saved to {model_path}')
print(f'  Results saved to {results_path}')
print('\nDone.')
