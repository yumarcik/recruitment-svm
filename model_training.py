import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

df = pd.read_csv("data.csv")
X = df[["experience_years", "technical_score"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

os.makedirs("model_files", exist_ok=True)
joblib.dump(model, "model_files/model.pkl")
joblib.dump(scaler, "model_files/scaler.pkl")

print("Model and scaler saved successfully.")