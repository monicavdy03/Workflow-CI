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

# pastikan target ada
if target_col not in df.columns:
    raise ValueError(f"ERROR: target '{target_col}' tidak ditemukan dalam dataset!")

# --- Drop kolom yang tidak berguna ---
if "Daily Summary" in df.columns:
    df = df.drop("Daily Summary", axis=1)

# --- Convert datetime ---
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors='coerce')
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    df = df.drop("time", axis=1)

# --- Encode kategorikal (Summary, Precip Type) ---
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna("Unknown")
    df[col] = LabelEncoder().fit_transform(df[col])

# --- Siapkan X dan y ---
X = df.drop(target_col, axis=1)
y = df[target_col]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
score = model.score(X_test, y_test)
print(f"✅ Model R^2 Score: {score}")

# --- Log MLflow ---
with mlflow.start_run():
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(model, "model")

print("Training complete and model logged to MLflow.")
