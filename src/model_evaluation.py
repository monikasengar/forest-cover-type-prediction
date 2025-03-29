import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data

# Define file paths
model_path = "models/random_forest.pkl"

def evaluate_model():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()