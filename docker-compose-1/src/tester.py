# src/tester.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def test_model():
    df = pd.read_csv("/app/data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load("/app/saved_models/iris_model.pkl")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    test_model()
