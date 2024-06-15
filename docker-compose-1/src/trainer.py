import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


def train_and_save_model():
    # Ensure the directories exist
    os.makedirs("/app/saved_models", exist_ok=True)

    # Load the dataset
    df = pd.read_csv("/app/data/iris.csv")  # Updated path
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "/app/saved_models/iris_model.pkl")
    print("Model saved to saved_models/iris_model.pkl")


if __name__ == "__main__":
    train_and_save_model()
