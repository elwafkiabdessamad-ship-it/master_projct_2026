import os, pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

DESC_NAMES = [d[0] for d in Descriptors._descList]
CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

EXAMPLE_MOLS = {
    "Ethanol (non-toxic)": "CCO",
    "Benzene (non-toxic)": "c1ccccc1",
    "Benzoic acid (non-toxic)": "O=C(O)c1ccccc1",
    "Phenol (non-toxic)": "Oc1ccccc1",
    "Methanol (non-toxic)": "CO",
    "Toluene (toxic)": "Cc1ccccc1",
    "Chloroform (non-toxic)": "ClC(Cl)Cl",
}


def load_pipeline(pkl_name: str) -> dict:
    path = os.path.join(MODELS_DIR, pkl_name)
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return None


def mol_to_image(smiles: str, size=(300, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def smiles_to_features(smiles: str, sel):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = np.array([GEN.GetFingerprint(mol)], dtype=np.float64)
    descs = np.array([list(CALCULATOR.CalcDescriptors(mol))], dtype=np.float64)
    X_raw = np.hstack([fp, descs])
    return sel.transform(X_raw)


def predict_toxicity(smiles: str, pipeline: dict, model_name: str):
    prep = pipeline["preprocessor"]
    X = smiles_to_features(smiles, prep["sel"])
    if X is None:
        return None
    model = pipeline["models"][model_name]
    if model_name in prep["needs_scaling"]:
        X = prep["scaler"].transform(X)
    proba = model.predict_proba(X)[0, 1]
    return int(proba > 0.5), float(proba)


def get_model_metrics(pipeline: dict, model_name: str) -> dict:
    for r in pipeline["results"]:
        if r["model"] == model_name:
            return r
    return {}


def get_feature_importance(pipeline: dict, model_name: str, top_n: int = 15):
    model = pipeline["models"].get(model_name)
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    feat_names = pipeline["preprocessor"]["feat_names"]
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feat_names[i], float(importances[i])) for i in idx]


INFO_TEXT = """### How it works

1. Enter a SMILES string
2. The molecule is featurised using:
   - **Morgan fingerprint** (radius=2, 2048 bits)
   - **217 RDKit descriptors**
3. Low-variance features are filtered (same as training)
4. The classifier predicts toxicity (log_LC50 < 1.0 → toxic)
"""
