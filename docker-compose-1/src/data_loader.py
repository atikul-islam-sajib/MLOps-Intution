import os
import pandas as pd
from sklearn.datasets import load_iris


def load_and_save_data():
    # Ensure the directory exists
    os.makedirs("/data", exist_ok=True)

    # Load the iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target

    # Save to CSV
    df.to_csv("/app/data/iris.csv", index=False)

    print("Data saved to /data/iris.csv")


if __name__ == "__main__":
    load_and_save_data()
