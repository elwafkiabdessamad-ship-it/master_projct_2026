# Abdou's QSAR Toxicity Prediction — Results

**Task:** Binary classification — predict toxic (log_LC50 < 1.0) vs non-toxic (log_LC50 ≥ 1.0)

---

## Dataset

| Molecules | Toxic | Non-toxic | % Toxic | logC50 range |
|-----------|-------|-----------|---------|--------------|
| 2,146     | 1,143 | 1,003     | 53.3%   | −5.00 to 4.76 |

**Featurisation:** Morgan fingerprint (2048 bits, radius 2) + 217 RDKit descriptors → VarianceThreshold (0.01) → 610 features retained.

**Deployment:** Gradio app (`app_abdou.py`)

---

## Classical ML Benchmark — 11 models

| Model | AUC-ROC | Accuracy | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|
| **RandomForest** | **0.8643** | **0.7884** | **0.8060** | 0.7875 | 0.8253 | 0.7463 |
| GradientBoosting | 0.8616 | 0.7581 | 0.7768 | 0.7637 | 0.7904 | 0.7214 |
| XGBoost | 0.8610 | 0.7860 | 0.8034 | 0.7866 | 0.8210 | 0.7463 |
| LightGBM | 0.8590 | 0.7721 | 0.7897 | 0.7764 | 0.8035 | 0.7363 |
| ExtraTrees | 0.8521 | 0.7581 | 0.7797 | 0.7572 | 0.8035 | 0.7065 |
| SVM | 0.8367 | 0.7581 | 0.7787 | 0.7593 | 0.7991 | 0.7114 |
| LogisticRegression | 0.7827 | 0.7279 | 0.7462 | 0.7414 | 0.7511 | 0.7015 |
| AdaBoost | 0.7808 | 0.7163 | 0.7426 | 0.7184 | 0.7686 | 0.6567 |
| KNN | 0.7590 | 0.6605 | 0.6524 | 0.7173 | 0.5983 | 0.7313 |
| GaussianNB | 0.7579 | 0.6605 | 0.6075 | 0.7902 | 0.4935 | 0.8507 |
| DecisionTree | 0.7186 | 0.7349 | 0.7729 | 0.7106 | 0.8472 | 0.6070 |

**Top 3:** RandomForest, GradientBoosting, XGBoost

---

## Transformer Comparison (ChemBERTa) vs Best Classical

| Model | AUC-ROC | Accuracy | F1 | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|
| ChemBERTa-Druglike | **0.8282** | 0.7093 | 0.7073 | 0.6594 | 0.7662 | 0.7626 |
| ChemBERTa-MLM | 0.8261 | 0.7535 | 0.7745 | 0.7948 | 0.7065 | 0.7552 |
| ChemBERTa-MTR | 0.8112 | 0.7488 | 0.7840 | 0.8559 | 0.6269 | 0.7232 |
| **Best classical** (RandomForest) | **0.8643** | **0.7884** | **0.8060** | 0.8253 | 0.7463 | 0.7875 |

**Finding:** Classical ML outperforms all ChemBERTa variants tested.

---

## Deployment App

| App | Framework | Best Model | AUC-ROC |
|---|---|---|---|
| `app_abdou.py` | Gradio | ExtraTrees | 0.895 |

**Features:** Morgan fingerprint (2048) + 217 RDKit descriptors → VarianceThreshold → classifier.
