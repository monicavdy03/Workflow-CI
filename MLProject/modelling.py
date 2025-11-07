import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Argument parser untuk file dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="weather_preprocessed.csv")
args = parser.parse_args()

# Muat dataset
data = pd.read_csv(args.data_path)
print("Dataset loaded:", data.shape)

# Pastikan kolom sesuai dataset kamu
X = data.drop(columns=["Temperature (C)"])
y = data["Temperature (C)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai MLflow run
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

print("✅ Model berhasil dilatih dengan MLflow Project.")
