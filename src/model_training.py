import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

# Define file paths
model_path = "models/random_forest.pkl"

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, "models/scaler.pkl")
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.2f}")
    
    return model, acc
