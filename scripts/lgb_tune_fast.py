"""
LightGBM — focused tuning with early stopping.
"""
import os, warnings, pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MolStandardize
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')
RNG = 42

DATA_IN  = 'data/the_final_data_base_in_KNIME.csv'
OUT_DIR  = 'models'
os.makedirs(OUT_DIR, exist_ok=True)

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

meta_cols = ['CAS_str', 'Chemical_name', 'SMILES', 'mlecule_H_final', 'SMILES_curated', 'mol', 'inorganic']
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
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=RNG)
print(f'Features: {X_final.shape[1]}, Tr: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

# ── Baseline ──
base = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                          random_state=RNG, n_jobs=-1, verbose=-1)
base.fit(X_tr, y_tr)
y_pred_base = base.predict(X_test)
base_r2 = r2_score(y_test, y_pred_base)
print(f'\nBaseline: R²={base_r2:.4f}')

# ── Sweep reg_alpha/reg_lambda ──
print('\n── Reg sweep (early stopping) ──')
reg_grid = [(0,0), (0.1,0.1), (0.5,0.5), (1,1), (2,2), (5,5), (0.5,1), (1,0.5)]
best_val, best_p, best_m = -1e9, None, None
for a, l in reg_grid:
    m = lgb.LGBMRegressor(n_estimators=2000, max_depth=7, learning_rate=0.01,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                           reg_alpha=a, reg_lambda=l, num_leaves=63,
                           random_state=RNG, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    vr2 = r2_score(y_val, m.predict(X_val))
    tr2 = r2_score(y_tr, m.predict(X_tr))
    n = m.best_iteration_
    print(f'  α={a:3.1f} λ={l:3.1f} n={n:4d} TrR²={tr2:.4f} ValR²={vr2:.4f} Gap={tr2-vr2:.4f}')
    if vr2 > best_val: best_val, best_p, best_m = vr2, (a, l), m

# ── Depth sweep ──
print('\n── Depth sweep ──')
best_val2, best_d, best_m2 = -1e9, None, None
for d in [5, 7, 10, 12, 15]:
    m = lgb.LGBMRegressor(n_estimators=2000, max_depth=d, learning_rate=0.01,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                           reg_alpha=best_p[0], reg_lambda=best_p[1], num_leaves=min(255, 2**d),
                           random_state=RNG, n_jobs=-1, verbose=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    vr2 = r2_score(X_val, y_val) if False else r2_score(y_val, m.predict(X_val))
    vr2 = r2_score(y_val, m.predict(X_val))
    tr2 = r2_score(y_tr, m.predict(X_tr))
    print(f'  depth={d:2d} n={m.best_iteration_:4d} TrR²={tr2:.4f} ValR²={vr2:.4f} Gap={tr2-vr2:.4f}')
    if vr2 > best_val2: best_val2, best_d, best_m2 = vr2, d, m

# ── Final: retrain on all train ──
print('\n── Final ──')
nl = min(255, 2**best_d)
final = lgb.LGBMRegressor(n_estimators=2000, max_depth=best_d, learning_rate=0.01,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
                           reg_alpha=best_p[0], reg_lambda=best_p[1], num_leaves=nl,
                           random_state=RNG, n_jobs=-1, verbose=-1)
final.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

y_pred = final.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
tr2 = r2_score(y_train, final.predict(X_train))
print(f'\n  LightGBM tuned: R²={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}')
print(f'  Train R²={tr2:.4f} Gap={tr2-r2:.4f}')
print(f'  Δ vs baseline: {r2 - base_r2:+.4f}')
print(f'  Best params: α={best_p[0]}, λ={best_p[1]}, depth={best_d}, leaves={nl}')

with open(os.path.join(OUT_DIR, 'LightGBM_tuned.pkl'), 'wb') as f:
    pickle.dump(final, f)
print('Saved models/LightGBM_tuned.pkl')
