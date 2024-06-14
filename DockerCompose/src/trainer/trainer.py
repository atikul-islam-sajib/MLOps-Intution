import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("/trainer/data/iris.csv")

# Split the data into features and target
X = data.drop(columns=["target"])
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure the "model" directory exists
model_dir = "/trainer/models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model to a file
with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Model training complete.")
