# QSAR Acute Aquatic Toxicity Prediction — Results Summary

**Task:** Binary classification — predict toxic (log_LC50 < 1.0) vs non-toxic (log_LC50 ≥ 1.0)

---

## 1. Dataset Overview

| Dataset | Molecules | Toxic | Non-toxic | % Toxic | logC50 range |
|---------|-----------|-------|-----------|---------|--------------|
| Imane   | 1,559     | 842   | 717       | 54.0%   | −4.38 to 4.97 |
| Abdou   | 2,146     | 1,143 | 1,003     | 53.3%   | −5.00 to 4.76 |

**Featurisation:** Morgan fingerprint (2048 bits, radius 2) + 217 RDKit descriptors → VarianceThreshold (0.01) filter.

---

## 2. Classical ML Benchmark — Imane dataset

| Model | AUC-ROC | Accuracy | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|
| GradientBoosting | **0.8241** | 0.7660 | 0.7754 | 0.8077 | 0.7456 | 0.7902 |
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

**Top 3 (by AUC-ROC):** GradientBoosting, ExtraTrees, RandomForest

---

## 3. Classical ML Benchmark — Abdou dataset

| Model | AUC-ROC | Accuracy | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|
| RandomForest | **0.8643** | 0.7884 | 0.8060 | 0.7875 | 0.8253 | 0.7463 |
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

**Top 3 (by AUC-ROC):** RandomForest, GradientBoosting, XGBoost

---

## 4. GNN Comparison (Imane split)

| Model | AUC-ROC | Accuracy | F1 | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|
| **SimpleGNN** (GATv2) | **0.7911** | 0.7308 | 0.7407 | 0.7101 | 0.7552 | 0.7742 |
| CGCNN | 0.7660 | 0.7147 | 0.7493 | 0.7870 | 0.6294 | 0.7151 |
| LSTM (SMILES) | 0.7322 | 0.6795 | 0.7076 | 0.7160 | 0.6364 | 0.6994 |
| **Best classical** (GradientBoosting) | **0.8241** | 0.7660 | 0.7754 | 0.7456 | 0.7902 | 0.8077 |

---

## 5. Transformer Comparison (ChemBERTa, Abdou split)

| Model | AUC-ROC | Accuracy | F1 | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|
| **ChemBERTa-Druglike** | **0.8282** | 0.7093 | 0.7073 | 0.6594 | 0.7662 | 0.7626 |
| ChemBERTa-MLM | 0.8261 | 0.7535 | 0.7745 | 0.7948 | 0.7065 | 0.7552 |
| ChemBERTa-MTR | 0.8112 | 0.7488 | 0.7840 | 0.8559 | 0.6269 | 0.7232 |
| **Best classical** (RandomForest) | **0.8643** | 0.7884 | 0.8060 | 0.8253 | 0.7463 | 0.7875 |

---

## 6. Best Models — Final Comparison

| Approach | Best Model | AUC-ROC | Accuracy | F1 |
|---|---|---|---|---|
| Classical ML (Imane) | GradientBoosting | 0.8241 | 0.7660 | 0.7754 |
| Classical ML (Abdou) | RandomForest | **0.8643** | **0.7884** | **0.8060** |
| GNN | SimpleGNN (GATv2) | 0.7911 | 0.7308 | 0.7407 |
| Transformer | ChemBERTa-Druglike | 0.8282 | 0.7093 | 0.7073 |

**Key finding:** Classical ML with Morgan fingerprints + RDKit descriptors outperforms both GNN and Transformer approaches on this dataset.

---

## 7. Deployment Apps

| App | Framework | Best Model | AUC-ROC |
|---|---|---|---|
| `app_imane.py` | Streamlit | LightGBM (top 3: GradientBoosting, ExtraTrees, RandomForest) | 0.832 |
| `app_abdou.py` | Gradio | ExtraTrees (top 3: RandomForest, GradientBoosting, XGBoost) | 0.895 |

Both apps use Morgan fingerprint (2048 bits) + 217 RDKit descriptors → VarianceThreshold → classifier.
