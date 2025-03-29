import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/train.csv")

# Separate features and target variable
X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the trained model inside the models folder
model_path = os.path.join("models", "random_forest.pkl")
joblib.dump(model, model_path)
print(f"Model saved as {model_path}")
