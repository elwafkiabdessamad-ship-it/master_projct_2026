"""
XGBoost refinement — second round with tighter grid + early stopping.
Focuses on learning_rate, n_estimators, and regularization params.
"""

import os, warnings, pickle, time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')
RNG = 42

# ── paths ──
DATA_IN  = 'data/the_final_data_base_in_KNIME.csv'
OUT_DIR  = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

# ========================================================================
# 1-3. DATA PIPELINE (same as round 1)
# ========================================================================
print('── Data pipeline ──')
df = pd.read_csv(DATA_IN)
meta = df['CAS_str'].astype(str).str.contains('Meta', na=False)
df = df[~meta].reset_index(drop=True)

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
    has_c = any(at.GetAtomicNum() == 6 for at in mol.GetAtoms())
    if not has_c:
        return True
    for at in mol.GetAtoms():
        if at.GetSymbol() not in organic_set:
            return True
    return False

df['mol'] = df['SMILES'].apply(parse_smiles)
df = df.dropna(subset=['mol']).reset_index(drop=True)
df['mol'] = df['mol'].apply(remove_salt)
uncharger = rdMolStandardize.Uncharger()
df['mol'] = df['mol'].apply(uncharger.uncharge)
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
df['inorganic'] = df['mol'].apply(is_inorganic)
df = df[~df['inorganic']].reset_index(drop=True)

meta_cols = ['CAS_str', 'Chemical_name', 'SMILES', 'mlecule_H_final',
             'SMILES_curated', 'mol', 'inorganic']
target_col = 'log_LC50'
desc_cols = [c for c in df.columns if c not in meta_cols + [target_col, 'n_measurements']]
X_desc = df[desc_cols].values
y = df[target_col].values
mols = df['mol'].tolist()
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(m) for m in mols]
X_fp = np.array(fps)
X_combined = np.hstack([X_desc, X_fp])
all_feature_names = desc_cols + [f'FP_{i}' for i in range(X_fp.shape[1])]

var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_combined)
kept_mask = var_thresh.get_support()
kept_names = [all_feature_names[i] for i in range(len(all_feature_names)) if kept_mask[i]]
corr_threshold = 0.95
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
keep_idx = [i for i in range(X_var.shape[1]) if i not in to_drop]
X_final = X_var[:, keep_idx]
final_names = [kept_names[i] for i in keep_idx]

print(f'  Final features: {X_final.shape[1]}, samples: {X_final.shape[0]}')

# ========================================================================
# 4. SPLIT  (further split train into train/val for early stopping)
# ========================================================================
print('── Split ──')
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=RNG
)
# Further split training into train/val for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RNG  # 0.2 * 0.8 = 0.16 of total
)
print(f'  Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

# ========================================================================
# 5. BASELINE  (validate our split matches notebook's default)
# ========================================================================
print('── Baseline ──')
baseline = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             random_state=RNG, n_jobs=-1, verbosity=0)
baseline.fit(X_tr, y_tr)
y_pred_base = baseline.predict(X_test)
base_r2 = r2_score(y_test, y_pred_base)
print(f'  Notebook-default XGBoost Test R²={base_r2:.4f}')

# ========================================================================
# 6. REFINED RANDOM SEARCH  (tighter grid around round-1 best)
# ========================================================================
print('── Refined random search (with early stopping) ──')

# We'll use the eval_set for early stopping inside each CV fold,
# but sklearn API + early stopping is tricky. Instead we use
# RandomizedSearchCV with a custom estimator that supports early stopping.
# Simpler approach: do a narrow grid search and use eval_set.

# Narrowed param grid around round-1 best
param_dist = {
    'n_estimators':       [400, 600, 800, 1000, 1200, 1500],
    'max_depth':          [6, 8, 10, 12],
    'learning_rate':      [0.005, 0.008, 0.01, 0.015, 0.02, 0.03],
    'subsample':          [0.6, 0.7, 0.8],
    'colsample_bytree':   [0.5, 0.6, 0.7],
    'min_child_weight':   [3, 5, 7, 10],
    'gamma':              [0.1, 0.2, 0.3],
    'reg_alpha':          [0.5, 1.0, 2.0, 5.0],
    'reg_lambda':         [0.5, 1.0, 2.0, 5.0],
}

xgb_model = xgb.XGBRegressor(
    random_state=RNG, n_jobs=-1, verbosity=0,
)
cv = KFold(n_splits=5, shuffle=True, random_state=RNG)

search = RandomizedSearchCV(
    xgb_model, param_dist, n_iter=80,
    scoring='r2', cv=cv, random_state=RNG, n_jobs=-1,
    verbose=1, return_train_score=True,
)

t0 = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - t0

best_params = search.best_params_
print(f'\n  Search completed in {elapsed:.1f}s')
print(f'  Best CV R²: {search.best_score_:.4f}')
print(f'  Best params: {best_params}')

# ── Refit best model with early stopping ──
print('\n── Refit with early stopping ──')
best = xgb.XGBRegressor(**best_params, random_state=RNG, n_jobs=-1,
                         verbosity=0, early_stopping_rounds=50,
                         eval_metric='rmse')
best.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

y_pred_best = best.predict(X_test)
best_r2  = r2_score(y_test, y_pred_best)
best_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
best_mae = mean_absolute_error(y_test, y_pred_best)

y_train_pred = best.predict(X_tr)
train_r2 = r2_score(y_tr, y_train_pred)

print(f'  Baseline  →  R²={base_r2:.4f}')
print(f'  Tuned     →  R²={best_r2:.4f}  RMSE={best_rmse:.4f}  MAE={best_mae:.4f}')
print(f'  Train R² (tuned): {train_r2:.4f}, Gap: {train_r2 - best_r2:.4f}')
print(f'  Δ R² vs baseline: {best_r2 - base_r2:+.4f}')
best_iter = best.best_iteration if hasattr(best, 'best_iteration') else 'N/A'
print(f'  Best iteration: {best_iter}')

# ========================================================================
# 7. SAVE
# ========================================================================
print('\n── Save ──')
model_path = os.path.join(OUT_DIR, 'XGBoost_tuned_r2.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best, f)

results = {
    'baseline': {'R2': base_r2, 'RMSE': None, 'MAE': None},
    'tuned':    {'R2': best_r2, 'RMSE': best_rmse, 'MAE': best_mae,
                 'train_R2': train_r2, 'gap': train_r2 - best_r2},
    'best_params': best_params,
    'cv_score_mean': search.best_score_,
    'cv_score_std': search.cv_results_['std_test_score'][search.best_index_],
}
results_path = os.path.join(OUT_DIR, 'XGBoost_tuning_results_r2.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print(f'  Model: {model_path}')
print('Done.')
