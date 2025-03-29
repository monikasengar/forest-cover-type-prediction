import joblib
import numpy as np

def predict_forest_type(features):
    model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    return int(prediction[0])

if __name__ == "__main__":
    sample_input = [2804, 139, 9, 268, 65, 3180, 234, 238, 135, 6121] + [0]*40
    print("Predicted Cover Type:", predict_forest_type(sample_input))
