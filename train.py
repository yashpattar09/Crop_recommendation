import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Crop_recommendation.csv")

# Features & labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# 2. SPLIT DATA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# 3. TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 4. EVALUATE MODEL
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Completed.")
print(f"Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# 5. SAVE MODEL
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the label encoder too (optional, in case future decoding needed)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("model.pkl saved successfully in the current directory!")
