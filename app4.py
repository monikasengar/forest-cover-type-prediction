import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
model_path = os.path.join("models", "random_forest.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index2.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get numerical feature values from the form
        features = [
            float(request.form["Elevation"]),
            float(request.form["Aspect"]),
            float(request.form["Slope"]),
            float(request.form["Horizontal_Distance_To_Hydrology"]),
            float(request.form["Vertical_Distance_To_Hydrology"]),
            float(request.form["Horizontal_Distance_To_Roadways"]),
            float(request.form["Hillshade_9am"]),
            float(request.form["Hillshade_Noon"]),
            float(request.form["Hillshade_3pm"]),
            float(request.form["Horizontal_Distance_To_Fire_Points"])
        ]

        # Wilderness Area (One-hot encoding)
        wilderness_selected = int(request.form["Wilderness_Area"])
        wilderness_vector = [1 if i == wilderness_selected else 0 for i in range(1, 5)]
        features.extend(wilderness_vector)

        # Soil Types (One-hot encoding)
        for i in range(1, 41):
            features.append(1 if f"Soil_Type{i}" in request.form else 0)

        # Ensure the correct number of features (55) by adding a placeholder if needed
        while len(features) < 55:
            features.append(0)

        # Debugging: Print feature length
        print(f"Feature length before prediction: {len(features)}")

        # Convert input to numpy array and reshape for model
        input_data = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("index2.html", prediction=prediction)

    except Exception as e:
        return str(e)

if __name__=="__main__":
    app.run(debug=True)