import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(path):
    return pd.read_csv(path)

def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    mlflow.start_run()

    data = load_data(args.data_path)
    model, acc = train_model(data)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()

    print(f"Training completed! Accuracy: {acc}")
