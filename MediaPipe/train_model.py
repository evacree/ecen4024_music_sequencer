import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"

df = pd.read_csv(CSV_PATH)

X = df.drop("label", axis=1)
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)



pred = model.predict(X_test)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
