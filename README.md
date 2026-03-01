# Early Detection of Student Burnout & Dropout Risk

## Problem Overview
Universities often detect student disengagement only after academic performance drops significantly. This project uses behavioural analytics to identify early signals of burnout and dropout risk based on academic engagement, progression, and performance indicators.

The system predicts:
- Burnout Risk Level (Low / Medium / High)
- Dropout Probability
- Key behavioural triggers influencing risk
- Explainable insights to support early intervention

---

## Dataset Description

### Dataset Type
Public Dataset

### Dataset Source
UCI Machine Learning Repository  
Dataset: Predict Students’ Dropout and Academic Success  
Link: https://archive.ics.uci.edu/ml/datasets/Predict+students%27+dropout+and+academic+success

### Why This Dataset Fits Behavioural Analytics
This dataset captures behavioural patterns related to:
- Academic engagement (course enrollments, approvals, grades)
- Attendance mode (daytime / evening)
- Socio-economic and demographic context
- Progression consistency across semesters

These attributes collectively reflect student behavioural engagement, persistence, and academic stress signals.

---

## Behavioural Features Used
Key behavioural indicators derived from the dataset include:
- Semester-wise course approval rates
- Grade consistency across semesters
- Enrollment vs approval gaps
- Age at enrollment and attendance mode
- Financial pressure indicators (debtor status, tuition fee status)
- Academic momentum indicators

---

## Project Structure

student_burnout_project  
├── data  
│   ├── student_behaviour_data.csv  
│   ├── processed_data.csv  
│   └── student_risk_scores.csv  
├── models  
│   ├── logistic_model.pkl  
│   └── random_forest_model.pkl  
├── src  
│   ├── preprocess.py  
│   ├── model.py  
│   ├── risk_scoring.py  
│   └── dashboard.py  
├── requirements.txt  
└── README.md

---

## Modelling Approach
- Logistic Regression for interpretable dropout probability estimation
- Random Forest for non-linear behavioural pattern learning
- Rule-based burnout risk categorization for explainability

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

---

## Dashboard
An interactive Streamlit dashboard visualizes:
- Burnout risk distribution
- Individual student risk probabilities
- Behavioural explanations for risk classification

---

## How to Run
pip install -r requirements.txt  
python src/preprocess.py  
python src/model.py  
python src/risk_scoring.py  
streamlit run src/dashboard.py
