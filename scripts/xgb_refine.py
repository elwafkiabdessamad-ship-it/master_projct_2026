"""
XGBoost — reduction of train-test gap via:
- early stopping with 20% validation split
- feature selection (top N by importance)
- lower learning rate + more trees
- L1/L2 regularization sweep
"""

import os, warnings, pickle, time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

warnings.filterwarnings('ignore')
RNG = 42

DATA_IN  = 'data/the_final_data_base_in_KNIME.csv'
OUT_DIR  = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data pipeline ──
df = pd.read_csv(DATA_IN)
meta = df['CAS_str'].astype(str).str.contains('Meta', na=False)
df = df[~meta].reset_index(drop=True)

organic_set = {'H','B','C','N','O','F','Si','P','S','Cl','Se','Br','I'}
def parse_smiles(s):
    try: return Chem.MolFromSmiles(s)
    except: return None
def remove_salt(mol):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) == 1: return mol
    largest = max(frags, key=lambda m: m.GetNumAtoms())
    try: Chem.SanitizeMol(largest); return largest
    except: return mol
def is_inorganic(mol):
    has_c = any(at.GetAtomicNum() == 6 for at in mol.GetAtoms())
    if not has_c: return True
    for at in mol.GetAtoms():
        if at.GetSymbol() not in organic_set: return True
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
    except: pass
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

var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_combined)
corr_threshold = 0.95
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
keep_idx = [i for i in range(X_var.shape[1]) if i not in to_drop]
X_final = X_var[:, keep_idx]

# Split: train/val/test (60/20/20)
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=RNG)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RNG)  # 0.25*0.8 = 0.2
print(f'Features: {X_final.shape[1]}')
print(f'Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

# ── Sweep over reg_alpha and reg_lambda with early stopping ──
print('\n── Regularization sweep (early stopping) ──')

base_params = {
    'n_estimators': 3000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.1,
}

reg_grid = [
    (0.0, 0.0), (0.1, 0.1), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0),
    (5.0, 5.0), (10.0, 10.0),
    (0.5, 1.0), (1.0, 0.5), (1.0, 2.0), (2.0, 1.0),
    (10.0, 0.0), (0.0, 10.0),
]

best_val_r2 = -1e9
best_reg = None
best_model = None

for reg_a, reg_l in reg_grid:
    p = {**base_params, 'reg_alpha': reg_a, 'reg_lambda': reg_l}
    m = xgb.XGBRegressor(**p, random_state=RNG, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=100, eval_metric='rmse')
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_r2 = r2_score(y_val, m.predict(X_val))
    train_r2 = r2_score(y_tr, m.predict(X_tr))
    n_used = m.best_iteration if hasattr(m, 'best_iteration') else m.config['n_estimators']
    print(f'  α={reg_a:5.1f}  λ={reg_l:5.1f}  n={n_used:4d}  '
          f'Train R²={train_r2:.4f}  Val R²={val_r2:.4f}  '
          f'Gap={train_r2 - val_r2:.4f}')
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_reg = (reg_a, reg_l)
        best_model = m

# ── Depth sweep ──
print('\n── Depth sweep (with best reg) ──')
best_depth_r2 = -1e9
best_depth = None
best_depth_model = None

for d in [4, 5, 6, 7, 8, 10, 12]:
    p = {**base_params, 'max_depth': d,
         'reg_alpha': best_reg[0], 'reg_lambda': best_reg[1]}
    m = xgb.XGBRegressor(**p, random_state=RNG, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=100, eval_metric='rmse')
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_r2 = r2_score(y_val, m.predict(X_val))
    train_r2 = r2_score(y_tr, m.predict(X_tr))
    n_used = m.best_iteration if hasattr(m, 'best_iteration') else p['n_estimators']
    print(f'  depth={d:2d}  n={n_used:4d}  Train R²={train_r2:.4f}  '
          f'Val R²={val_r2:.4f}  Gap={train_r2 - val_r2:.4f}')
    if val_r2 > best_depth_r2:
        best_depth_r2 = val_r2
        best_depth = d
        best_depth_model = m

# ── Final: retrain best config on train+val ──
print('\n── Final model (best config, train on all train data) ──')
final_params = {
    'n_estimators': 3000,
    'max_depth': best_depth,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': best_reg[0],
    'reg_lambda': best_reg[1],
}
print(f'  Best α={best_reg[0]}, λ={best_reg[1]}, depth={best_depth}')

final = xgb.XGBRegressor(**final_params, random_state=RNG, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=100, eval_metric='rmse')
final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred = final.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
train_r2_final = r2_score(y_train, final.predict(X_train))
final_gap = train_r2_final - test_r2
best_iter = final.best_iteration if hasattr(final, 'best_iteration') else 'N/A'

# Baseline
base = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         random_state=RNG, n_jobs=-1, verbosity=0)
base.fit(X_tr, y_tr)
y_pred_base = base.predict(X_test)
base_r2 = r2_score(y_test, y_pred_base)

print(f'\n── Results ──')
print(f'  Baseline:                R²={base_r2:.4f}')
print(f'  Tuned (n={best_iter}):   R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}')
print(f'  Train R²={train_r2_final:.4f}  Gap={final_gap:.4f}')
print(f'  Δ R²: {test_r2 - base_r2:+.4f}')

# Save
with open(os.path.join(OUT_DIR, 'XGBoost_tuned_final.pkl'), 'wb') as f:
    pickle.dump(final, f)

results = {
    'test_R2': test_r2, 'test_RMSE': test_rmse, 'test_MAE': test_mae,
    'train_R2': train_r2_final, 'gap': final_gap,
    'baseline_R2': base_r2,
    'best_params': final_params,
    'best_iteration': best_iter,
}
with open(os.path.join(OUT_DIR, 'XGBoost_tuned_final_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

print(f'\nSaved models/XGBoost_tuned_final.pkl')
print('Done.')
