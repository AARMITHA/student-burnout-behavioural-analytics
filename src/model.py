import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_models(data_path):
    df = pd.read_csv(data_path)

    X = df.drop(columns=["burnout_risk"])
    y = df["burnout_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ])

    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))

    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))

    os.makedirs("../models", exist_ok=True)

    joblib.dump(lr_pipeline, "../models/logistic_model.pkl")
    joblib.dump(rf, "../models/random_forest_model.pkl")

    print("Models saved successfully")

if __name__ == "__main__":
    train_models("../data/processed_data.csv")