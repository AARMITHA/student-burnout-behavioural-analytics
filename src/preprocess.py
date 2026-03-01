import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if len(df.columns) == 1:
        df = pd.read_csv(file_path, encoding="utf-8-sig", sep=",")

    df.columns = [c.strip() for c in df.columns]

    target_col = next(c for c in df.columns if c.lower() == "target")

    required_cols = [
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Tuition fees up to date"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    categorical_cols.remove(target_col)

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df[target_col] = le.fit_transform(df[target_col].astype(str))

    df["burnout_risk"] = (
        (df["Curricular units 1st sem (approved)"] == 0) &
        (df["Curricular units 2nd sem (approved)"] == 0) &
        (df["Tuition fees up to date"] == 0)
    ).astype(int)

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("../data/student_behaviour_data.csv")
    df.to_csv("../data/processed_data.csv", index=False)
    print("Preprocessing completed successfully")