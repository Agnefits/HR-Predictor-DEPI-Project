# Employee Attrition Prediction

## Team Information

**Names:**
- Abdallah Salah  
- Ahmed Adel  
- Morad Ahmed  
- Shahd Mohamed  
- Youssef Ahmed  

**Group Code:** AST4_AIS2_S1  

**Instructor:** Eng. Mahmoud Talaat  

**Program:** DEPI – Round 4  

**Track:** AI & Data Science – Microsoft Machine Learning Engineer

## Introduction
Employee attrition is a major challenge for organizations. 
This project uses Machine Learning and Deep Learning to predict 
whether an employee will leave the company based on HR data.

## Problem Statement
Given HR data including satisfaction level, salary, department, 
and work hours, we aim to classify whether an employee will leave (1) or stay (0).

- **Problem Type:** Binary Classification  
- **Target Variable:** `left`  
- **Success Metric:** F1 Score, ROC-AUC

## Dataset Description
- **Source:** HR_Dataset.csv  
- **Size:** ~15,000 rows  
- **Features:**
  - `satisfaction_level` – Employee satisfaction score
  - `last_evaluation` – Last performance review score
  - `number_project` – Number of projects assigned
  - `average_montly_hours` – Average monthly working hours
  - `time_spend_company` – Years at company
  - `Work_accident` – Whether employee had a work accident
  - `promotion_last_5years` – Promoted in last 5 years
  - `Departments` – Department name
  - `salary` – Salary level (low / medium / high)

## Project Structure
project/
├── data/
│   └── HR_Dataset.csv
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── README.md
├── report.md
└── requirements.txt

## Methodology
1. Data Cleaning (handle missing values, duplicates, fix types)
2. EDA (distributions, correlation heatmap, attrition by department)
3. Preprocessing (ordinal encoding, one-hot encoding, robust scaling)
4. Model Building (Logistic Regression, Random Forest, XGBoost, ANN)
5. Model Optimization (RandomizedSearchCV + feature selection)
6. Final Model Selection

## Results
| Model | F1 Score | ROC-AUC |
|---|---|---|
| Logistic Regression | ~0.57 | ~0.83 |
| Random Forest (Tuned) | **0.9547** | **0.9793** |
| XGBoost | ~0.92 | ~0.97 |
| ANN | ~0.90 | ~0.96 |

## Conclusion
The **tuned Random Forest** model achieved the best performance 
with F1 = 0.9547 and ROC-AUC = 0.9793. 
Feature selection reduced features from 19 to 6 with no loss in performance.