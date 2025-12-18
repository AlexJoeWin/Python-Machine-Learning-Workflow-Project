# 🧠 ML Pipeline: from raw data to explainable predictions

This repository showcases a complete machine learning workflow, from data ingestion and cleaning to model selection, hyperparameter tuning, and explainability. The project emphasizes best practices in preprocessing, model evaluation, and interpretability using modern tools like SHAP and Optuna.

## 🔍 Workflow Overview

### 1. Data Inspection & Cleaning
- Renamed headers for clarity
- Verified and corrected data types
- Converted `"?"` strings to `NaN` and imputed missing values
- Identified rows with multiple `"?"` entries using a custom algorithm
- Corrected suspected typos in categorical features
- Checked for duplicates and high correlation among numeric features

### 2. Preprocessing
- **Target variable**: Label Encoding
- **Categorical features**: One-Hot Encoding + Mode Imputation
- **Numerical features**: Power Transformation + Mean Imputation
- Split into training and test sets

### 3. Modeling
- Models used:
  - `RandomForestClassifier`
  - `XGBClassifier`
  - `VotingClassifier` (Logistic Regression + SVM)
- Evaluation via `Cross-Validation` using `ROC_AUC_SCORE`
- Hyperparameter tuning:
  - `RandomizedSearchCV` for VotingClassifier
  - `Optuna` for RandomForest and XGBoost

### 4. Feature Importance & Explainability
- `.feature_importances_` and Permutation Importance
- SHAP values for interpretability
- Synergy analysis between top features across methods

### 5. Dimensionality Reduction
- PCA applied to reduce dimensionality while retaining 90% variance
- Re-trained models on reduced feature space
- Visual inspection of explained variance


