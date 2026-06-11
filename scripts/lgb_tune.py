"""
LightGBM tuning for QSAR log_LC50 prediction.
"""

import os, warnings, pickle, time
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')
RNG = 42

DATA_IN  = 'data/the_final_data_base_in_KNIME.csv'
OUT_DIR  = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data pipeline (same as before) ──
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
corr_matrix = pd.DataFrame(X_var).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
keep_idx = [i for i in range(X_var.shape[1]) if i not in to_drop]
X_final = X_var[:, keep_idx]

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=RNG)
print(f'Features: {X_final.shape[1]}, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}')

# ── Baseline (notebook defaults) ──
print('\n── Baseline (notebook defaults) ──')
base = lgb.LGBMRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    random_state=RNG, n_jobs=-1, verbose=-1,
)
base.fit(X_train, y_train)
y_pred_base = base.predict(X_test)
base_r2 = r2_score(y_test, y_pred_base)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
print(f'  R²={base_r2:.4f}  RMSE={base_rmse:.4f}')

# ── Randomized search ──
print('\n── Random search ──')
param_dist = {
    'n_estimators':       [200, 400, 600, 800, 1000],
    'max_depth':          [3, 5, 7, 10, 12, 15, -1],
    'learning_rate':      [0.01, 0.03, 0.05, 0.07, 0.1],
    'subsample':          [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':   [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples':  [5, 10, 20, 30, 50],
    'reg_alpha':          [0, 0.01, 0.1, 1.0, 5.0],
    'reg_lambda':         [0, 0.01, 0.1, 1.0, 5.0],
    'num_leaves':         [31, 63, 127, 255],
}

model = lgb.LGBMRegressor(random_state=RNG, n_jobs=-1, verbose=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
search = RandomizedSearchCV(
    model, param_dist, n_iter=80,
    scoring='r2', cv=cv, random_state=RNG, n_jobs=-1,
    verbose=1, return_train_score=True,
)

t0 = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - t0

best_params = search.best_params_
print(f'\n  Search: {elapsed:.1f}s')
print(f'  Best CV R²: {search.best_score_:.4f}')
print(f'  Best params: {best_params}')

# ── Evaluate ──
best = search.best_estimator_
y_pred = best.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
train_pred = best.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
gap = train_r2 - r2
print(f'\n  Test:  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}')
print(f'  Train: R²={train_r2:.4f}  Gap={gap:.4f}')
print(f'  Δ R² vs baseline: {r2 - base_r2:+.4f}')

# ── Save ──
with open(os.path.join(OUT_DIR, 'LightGBM_tuned.pkl'), 'wb') as f:
    pickle.dump(best, f)
print('\nSaved models/LightGBM_tuned.pkl')
print('Done.')
