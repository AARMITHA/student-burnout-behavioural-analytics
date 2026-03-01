import pandas as pd
import joblib
import numpy as np

def load_models():
    lr = joblib.load("../models/logistic_model.pkl")
    rf = joblib.load("../models/random_forest_model.pkl")
    return lr, rf

def risk_level(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"

def explain_risk(row):
    reasons = []

    if row["Curricular units 1st sem (approved)"] == 0:
        reasons.append("No subjects cleared in first semester")

    if row["Curricular units 2nd sem (approved)"] == 0:
        reasons.append("No subjects cleared in second semester")

    if row["Debtor"] == 1:
        reasons.append("Outstanding tuition fees")

    if row["Tuition fees up to date"] == 0:
        reasons.append("Tuition fees not up to date")

    if row["Age at enrollment"] > 30:
        reasons.append("Late age at enrollment")

    if len(reasons) == 0:
        reasons.append("Stable academic and financial indicators")

    return ", ".join(reasons)

def score_students(data_path):
    df = pd.read_csv(data_path)
    lr, rf = load_models()

    X = df.drop(columns=["burnout_risk"])

    prob_lr = lr.predict_proba(X)[:, 1]
    prob_rf = rf.predict_proba(X)[:, 1]

    df["burnout_probability"] = (prob_lr + prob_rf) / 2
    df["risk_category"] = df["burnout_probability"].apply(risk_level)
    df["risk_explanation"] = df.apply(explain_risk, axis=1)

    df.to_csv("../data/student_risk_scores.csv", index=False)
    print("Risk scoring completed successfully")

if __name__ == "__main__":
    score_students("../data/processed_data.csv")