import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
import sys

# ===========================
# Argument parser untuk file dataset
# ===========================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="weather_preprocessed.csv")
args = parser.parse_args()

# Pastikan file dataset ada
if not os.path.exists(args.data_path):
    print(f"❌ File dataset '{args.data_path}' tidak ditemukan di {os.getcwd()}")
    sys.exit(1)

# ===========================
# Muat dataset
# ===========================
data = pd.read_csv(args.data_path)
print(f"✅ Dataset loaded: {data.shape} dari {args.data_path}")

# Pastikan kolom target ada
target_col = "Temperature (C)"
if target_col not in data.columns:
    print(f"❌ Kolom target '{target_col}' tidak ditemukan. Kolom tersedia: {list(data.columns)}")
    sys.exit(1)

# ===========================
# Split data
# ===========================
X = data.drop(columns=[target_col])
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# MLflow Tracking
# ===========================
mlflow.set_experiment("Prediksi_Suhu")

with mlflow.start_run(run_name="LinearRegression_CI"):
    mlflow.sklearn.autolog()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    mlflow.sklearn.log_model(model, "model")

    print("📊 Metrics:")
    print(f"  - MAE: {mae:.3f}")
    print(f"  - MSE: {mse:.3f}")
    print(f"  - R2 : {r2:.3f}")

print("✅ Model berhasil dilatih dan dicatat di MLflow.")
