import os
import sys
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["target"] = iris.target

# Check if the directory exists, if not, create it
if not os.path.exists("/data_loader/data"):
    os.makedirs("/data_loader/data")

# Save the dataframe to CSV
data.to_csv(os.path.join("/data_loader/data", "iris.csv"), index=False)

print("Data loading complete.")
