"""
Save all 11 classical ML models + preprocessors as .pkl files for sharing.
Students can load these directly without retraining.
"""
import pandas as pd, numpy as np, pickle, os, warnings
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb
warnings.filterwarnings('ignore')
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_classical_models(data_csv, out_name):
    print(f"\n{'='*60}")
    print(f"Processing: {data_csv}")
    print(f"{'='*60}")

    df = pd.read_csv(data_csv)
    df.rename(columns={'logC50': 'log_LC50', 'SMILES_curated': 'SMILES'}, inplace=True)
    meta = df['CAS'].astype(str).str.contains('Meta', na=False)
    df = df[~meta].reset_index(drop=True)
    df['mol'] = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) if s else None)
    df = df.dropna(subset=['mol']).reset_index(drop=True)
    df['toxic'] = (df['log_LC50'] < 1.0).astype(int)
    print(f"Molecules: {len(df)} | Toxic: {df['toxic'].sum()} ({df['toxic'].mean()*100:.1f}%)")

    from sklearn.model_selection import train_test_split
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = np.array([gen.GetFingerprint(m) for m in df['mol']])
    desc_names = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    descs = np.array([list(calculator.CalcDescriptors(m)) for m in df['mol']])
    X_raw = np.hstack([fps, descs])
    y = df['toxic'].values

    feat_names = np.array([f'Morgan_{i}' for i in range(2048)] + desc_names)

    sel = VarianceThreshold(0.01)
    X = sel.fit_transform(X_raw)
    print(f"Features: {X_raw.shape[1]} -> {X.shape[1]} after variance filter")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'GaussianNB': GaussianNB(),
    }

    results = []
    trained = {}
    for name, m in models.items():
        use_s = name in ['LogisticRegression', 'SVM', 'KNN', 'GaussianNB']
        X_tr, X_te = (X_train_s, X_test_s) if use_s else (X_train, X_test)
        m.fit(X_tr, y_train)
        if hasattr(m, 'predict_proba'):
            proba = m.predict_proba(X_te)[:, 1]
        else:
            proba = m.decision_function(X_te)
        pred = (proba > 0.5).astype(int)
        results.append({'model': name,
            'auc_roc': roc_auc_score(y_test, proba),
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'specificity': recall_score(1 - y_test, 1 - pred, zero_division=0)})
        print(f"  {name:25s} AUC={results[-1]['auc_roc']:.4f}")
        trained[name] = m

    dfr = pd.DataFrame(results).sort_values('auc_roc', ascending=False).reset_index(drop=True)
    top3 = dfr.head(3)['model'].tolist()
    print(f"\nTop 3: {top3}")

    pipeline = {
        'models': trained,
        'results': results,
        'top3': top3,
        'preprocessor': {
            'sel': sel,
            'scaler': scaler,
            'feat_names': feat_names[sel.get_support()],
            'needs_scaling': ['LogisticRegression', 'SVM', 'KNN', 'GaussianNB'],
        },
        'data_info': {
            'n_molecules': len(df),
            'n_toxic': int(df['toxic'].sum()),
            'n_non_toxic': int((1 - df['toxic']).sum()),
            'n_features_raw': X_raw.shape[1],
            'n_features_selected': X.shape[1],
        }
    }

    out_path = os.path.join(ROOT, 'models', out_name)
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Saved: {out_path}")
    return top3

# Save both variants
save_classical_models(
    os.path.join(ROOT, 'data', 'curated_data_rdkit_Imane.csv'),
    'classical_models_Imane.pkl')
save_classical_models(
    os.path.join(ROOT, 'data', 'curated_data_rdkit_Abdou.csv'),
    'classical_models_Abdou.pkl')

print("\nDone! Classical models saved.")
