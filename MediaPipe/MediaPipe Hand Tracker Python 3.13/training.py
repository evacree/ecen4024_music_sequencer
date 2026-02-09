import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("data/gesture_data.csv")

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model to a file
with open("data/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)