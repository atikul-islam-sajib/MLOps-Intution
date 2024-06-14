import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv("/tester/data/iris.csv")

# Split the data into features and target
X = data.drop(columns=["target"])
y = data["target"]

# Load the trained model
with open("/tester/models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict and evaluate the model
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"Model accuracy: {accuracy:.2f}")
