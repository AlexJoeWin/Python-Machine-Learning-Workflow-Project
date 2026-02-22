# 🧠 ML Pipeline: from raw data to explainable predictions

This repository showcases a complete machine learning workflow, from data ingestion and cleaning to model selection, hyperparameter tuning, and explainability. The project emphasizes best practices in preprocessing, model evaluation, and interpretability using modern tools like SHAP and Optuna.

## 🔍 Workflow Overview

### 1. Data Inspection & Cleaning
- Renamed headers for clarity
- Verified and corrected data types
- Converted `"?"` strings to `NaN` for numeric feature
- Split data into a numeric and string partition for better handling
- Determined unique categorical values and corrected typos
- Identified fraction of `"?"` in each feature, as well as rows with multiple `"?"` entries
- Converted `"?"` strings to `NaN` for all feature
- Statistics on numeric features
- Checked for high correlation among numeric features
- Checked for duplicate rows
- Transformed string features into categories
- Checked class balance

### 2. Preprocessing
- Split into training and test sets
- **Target variable**: Label Encoding
- **Categorical features**: One-Hot Encoding + Mode Imputation
- **Numerical features**: Power Transformation + Mean Imputation

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
- Synergy analysis between top features across methods
- SHAP values for interpretability

### 5. Dimensionality Reduction
- PCA applied to reduce dimensionality while retaining 90% variance
- Visual inspection of explained variance
- Re-trained models on reduced feature space
