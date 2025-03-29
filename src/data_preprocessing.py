import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define file paths
data_path = "data/train.csv"

def load_and_preprocess_data():
    df = pd.read_csv(data_path)
    
    # Drop ID column
    df = df.drop(columns=["Id"])
    
    # Splitting features and target
    X = df.drop(columns=["Cover_Type"])
    y = df["Cover_Type"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardization (optional)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
