import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import argparse
import os

# Argument dari MLflow
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="weather_preprocessed.csv")
args = parser.parse_args()

print(f"📂 Loading dataset from: {args.data_path}")

# Path absolut ke file dataset
data_path = os.path.join(os.path.dirname(__file__), args.data_path)

df = pd.read_csv(data_path)
print("✔ Dataset loaded successfully!")
print(df.head())

# Pemisahan fitur dan target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"🔎 Model accuracy: {score}")

# Log ke MLflow
with mlflow.start_run():
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("accuracy", score)
    mlflow.sklearn.log_model(model, "model")

print("🎉 Training complete and model logged to MLflow!")
