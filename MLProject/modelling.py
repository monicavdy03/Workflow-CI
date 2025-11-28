import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os
import argparse

# --- Ambil argumen dari MLflow Project ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="weather_preprocessed.csv")
args = parser.parse_args()

print(f"Loading dataset from: {args.data_path}")

# --- Load dataset ---
data_path = os.path.join(os.path.dirname(__file__), args.data_path)
df = pd.read_csv(data_path)
print("Dataset columns:", df.columns.tolist())

# --- Target variable ---
target_col = "temperature"

if target_col not in df.columns:
    raise ValueError(f"ERROR: target '{target_col}' tidak ditemukan dalam dataset!")

# --- Drop kolom tidak dipakai ---
drop_cols = ["Daily Summary"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# --- Convert datetime ---
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["time"])

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour

    df = df.drop("time", axis=1)

# --- Encode kategorikal ---
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Prepare X dan y ---
X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
score = model.score(X_test, y_test)
print(f"✅ Model R^2 Score: {score}")

# --- Log ke MLflow ---
with mlflow.start_run():
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(model, "model")

print("Training complete and model logged to MLflow.")
