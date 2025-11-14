import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model trained & saved as crop_model.pkl")
