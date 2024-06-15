# src/inference.py
import joblib
import time
import numpy as np


def make_inference(new_data):
    time.sleep(1)
    model = joblib.load("/app/saved_models/iris_model.pkl")
    prediction = model.predict(new_data)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example data
    make_inference(new_data)
