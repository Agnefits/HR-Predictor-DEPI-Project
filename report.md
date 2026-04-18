# Project Report: Employee Attrition Prediction

## 1. Problem Definition
- **Type:** Binary Classification
- **Goal:** Predict if an employee will leave the company
- **Input:** HR features (satisfaction, salary, department, etc.)
- **Output:** `left` → 0 (Stayed) or 1 (Left)
- **Metrics:** F1 Score (primary), ROC-AUC (secondary)

---

## 2. Data Cleaning
- Stripped column name spaces
- Filled missing numeric values with **median**
- Filled missing categorical values with **mode**
- Removed duplicate rows
- Fixed data types using `pd.to_numeric()`

---

## 3. Exploratory Data Analysis (EDA)

### Key Findings:
- Employees with **low satisfaction** are more likely to leave
- Departments like **HR and Accounting** show higher attrition rates
- Employees working **very long or very short hours** tend to leave more
- **Low salary** employees have the highest attrition

---

## 4. Data Preprocessing
- **Ordinal Encoding:** `salary` → low=1, medium=2, high=3
- **One-Hot Encoding:** `Departments` column
- **RobustScaler:** applied to `average_montly_hours`, 
  `time_spend_company`, `number_project`
- **Train/Test Split:** 80% train / 20% test (stratified)

---

## 5. Model Building

### Models Used:
| Model | Key Settings |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=1000` |
| Random Forest | `n_estimators=300`, `max_depth=10`, `class_weight='balanced_subsample'` |
| XGBoost | `scale_pos_weight=ratio`, `n_estimators=300`, `learning_rate=0.05` |
| ANN | 3 layers (64→32→1), ReLU, Sigmoid, EarlyStopping |

---

## 6. Model Optimization

### Feature Selection:
- Used Random Forest feature importances
- Selected features with importance > 0.01
- Reduced from 19 → 6 features

### Hyperparameter Tuning:
- Used `RandomizedSearchCV` with 5-fold CV
- Scoring metric: F1
- 30 iterations explored

---

## 7. Final Results

| Model | F1 Score | ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.57 | ~0.83 |
| Random Forest (Tuned) | **0.9547** | **0.9793** |
| XGBoost | ~0.92 | ~0.97 |
| ANN | ~0.90 | ~0.96 |

---

## 8. Final Model: Random Forest (Tuned)

**Why Random Forest?**
- Highest F1 Score → best balance of Precision & Recall
- Strong ROC-AUC → excellent class separation
- More interpretable than ANN
- Feature selection worked perfectly (19 → 6 features, no loss)

---

## 9. Conclusion

The Random Forest model with hyperparameter tuning and feature 
selection is the best model for this problem. 

**Future improvements:**
- Try SMOTE for handling class imbalance
- Experiment with LightGBM
- Deploy the model as a web app using Flask or Streamlit