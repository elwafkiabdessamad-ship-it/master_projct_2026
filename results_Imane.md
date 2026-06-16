# Imane's QSAR Toxicity Prediction — Results

**Task:** Binary classification — predict toxic (log_LC50 < 1.0) vs non-toxic (log_LC50 ≥ 1.0)

---

## Dataset

| Molecules | Toxic | Non-toxic | % Toxic | logC50 range |
|-----------|-------|-----------|---------|--------------|
| 1,559     | 842   | 717       | 54.0%   | −4.38 to 4.97 |

**Featurisation:** Morgan fingerprint (2048 bits, radius 2) + 217 RDKit descriptors → VarianceThreshold (0.01) → 602 features retained.

**Deployment:** Streamlit app (`app_imane.py`)

---

## Classical ML Benchmark — 11 models

| Model | AUC-ROC | Accuracy | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|
| **GradientBoosting** | **0.8241** | **0.7660** | **0.7754** | **0.8077** | 0.7456 | **0.7902** |
| ExtraTrees | 0.8144 | 0.7564 | 0.7683 | 0.7925 | 0.7456 | 0.7692 |
| RandomForest | 0.8102 | 0.7404 | 0.7553 | 0.7716 | 0.7396 | 0.7413 |
| LightGBM | 0.8074 | 0.7179 | 0.7233 | 0.7718 | 0.6805 | 0.7622 |
| XGBoost | 0.8067 | 0.7083 | 0.7200 | 0.7500 | 0.6923 | 0.7273 |
| SVM | 0.8008 | 0.7500 | 0.7636 | 0.7826 | 0.7456 | 0.7552 |
| AdaBoost | 0.7706 | 0.7308 | 0.7391 | 0.7778 | 0.7041 | 0.7622 |
| KNN | 0.7605 | 0.6603 | 0.6395 | 0.7520 | 0.5562 | 0.7832 |
| LogisticRegression | 0.7360 | 0.7308 | 0.7470 | 0.7607 | 0.7337 | 0.7273 |
| GaussianNB | 0.7241 | 0.6346 | 0.6096 | 0.7236 | 0.5266 | 0.7622 |
| DecisionTree | 0.6666 | 0.6955 | 0.7003 | 0.7500 | 0.6568 | 0.7413 |

**Top 3:** GradientBoosting, ExtraTrees, RandomForest

---

## GNN Comparison vs Best Classical

| Model | AUC-ROC | Accuracy | F1 | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|
| SimpleGNN (GATv2) | **0.7911** | 0.7308 | 0.7407 | 0.7101 | 0.7552 | 0.7742 |
| CGCNN | 0.7660 | 0.7147 | 0.7493 | 0.7870 | 0.6294 | 0.7151 |
| LSTM (SMILES) | 0.7322 | 0.6795 | 0.7076 | 0.7160 | 0.6364 | 0.6994 |
| **Best classical** (GradientBoosting) | **0.8241** | **0.7660** | **0.7754** | 0.7456 | **0.7902** | **0.8077** |

**Finding:** Classical ML outperforms all GNN architectures tested.

---

## Deployment App

| App | Framework | Best Model | AUC-ROC |
|---|---|---|---|
| `app_imane.py` | Streamlit | LightGBM | 0.832 |

**Features:** Morgan fingerprint (2048) + 217 RDKit descriptors → VarianceThreshold → classifier.
