# Forest Cover Type Prediction

This project predicts the type of forest cover based on environmental factors such as elevation, slope, and soil type.

## Project Structure
- `data_preprocessing.py`: Loads and preprocesses data
- `model_training.py`: Trains the RandomForest model
- `model_evaluation.py`: Evaluates the trained model
- `predict.py`: Predicts forest cover type from input features
- `EDA.ipynb`: Exploratory Data Analysis
- `requirements.txt`: List of dependencies

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python model_training.py
   ```
3. Evaluate the model:
   ```bash
   python model_evaluation.py
   ```
4. Make a prediction:
   ```bash
   python predict.py