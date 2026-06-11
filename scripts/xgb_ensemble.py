"""
XGBoost — best params trained on full training set,
plus a multi-seed ensemble for variance reduction.
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

var_thresh = VarianceThreshold(threshold=0.01)
X_var = var_thresh.fit_transform(X_combined)
corr_threshold = 0.95
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
keep_idx = [i for i in range(X_var.shape[1]) if i not in to_drop]
X_final = X_var[:, keep_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=RNG
)
print(f'Features: {X_final.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# ── Best params from round 2 ──
best_params = {
    'subsample': 0.8,
    'reg_lambda': 0.5,
    'reg_alpha': 0.5,
    'n_estimators': 1000,
    'min_child_weight': 5,
    'max_depth': 12,
    'learning_rate': 0.005,
    'gamma': 0.1,
    'colsample_bytree': 0.5,
}
n_ensemble = 10

# ── Single best model ──
print('── Single best model ──')
best = xgb.XGBRegressor(**best_params, random_state=RNG, n_jobs=-1, verbosity=0)
best.fit(X_train, y_train)

y_pred_single = best.predict(X_test)
r2_single  = r2_score(y_test, y_pred_single)
rmse_single = np.sqrt(mean_squared_error(y_test, y_pred_single))
mae_single = mean_absolute_error(y_test, y_pred_single)
y_train_pred = best.predict(X_train)
gap = r2_score(y_train, y_train_pred) - r2_single

print(f'  R²={r2_single:.4f}  RMSE={rmse_single:.4f}  MAE={mae_single:.4f}  Gap={gap:.4f}')

# ── Multi-seed ensemble ──
print(f'── {n_ensemble}-seed ensemble ──')
ensemble_preds = np.zeros((n_ensemble, len(y_test)))
train_preds = np.zeros((n_ensemble, len(y_train)))

for i in range(n_ensemble):
    seed = RNG + i * 11
    m = xgb.XGBRegressor(**best_params, random_state=seed, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train)
    ensemble_preds[i] = m.predict(X_test)
    train_preds[i] = m.predict(X_train)

y_pred_ens = ensemble_preds.mean(axis=0)
r2_ens  = r2_score(y_test, y_pred_ens)
rmse_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))
mae_ens = mean_absolute_error(y_test, y_pred_ens)

y_train_ens = train_preds.mean(axis=0)
gap_ens = r2_score(y_train, y_train_ens) - r2_ens

print(f'  R²={r2_ens:.4f}  RMSE={rmse_ens:.4f}  MAE={mae_ens:.4f}  Gap={gap_ens:.4f}')
print(f'  Δ R² single→ensemble: {r2_ens - r2_single:+.4f}')

# ── Per-seed spread ──
seed_r2s = [r2_score(y_test, ensemble_preds[i]) for i in range(n_ensemble)]
print(f'  Per-seed R²: mean={np.mean(seed_r2s):.4f} ± {np.std(seed_r2s):.4f}  '
      f'range=[{min(seed_r2s):.4f}, {max(seed_r2s):.4f}]')

# ── Compare with notebook default ──
baseline = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             random_state=RNG, n_jobs=-1, verbosity=0)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
r2_base = r2_score(y_test, y_pred_base)
print(f'\n── Summary ──')
print(f'  Notebook baseline:       R²={r2_base:.4f}')
print(f'  Tuned single:            R²={r2_single:.4f}  Δ={r2_single - r2_base:+.4f}')
print(f'  Tuned {n_ensemble}-seed ensemble: R²={r2_ens:.4f}  Δ={r2_ens - r2_base:+.4f}')

# ── Save ──
with open(os.path.join(OUT_DIR, 'XGBoost_ensemble.pkl'), 'wb') as f:
    pickle.dump({'models': [best], 'ensemble': None, 'params': best_params}, f)

print('\nDone.')
