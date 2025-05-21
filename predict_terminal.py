import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import joblib
import mlflow.xgboost

def get_user_input():
    print("Enter passenger information:")
    Pclass = int(input("Pclass (1, 2, or 3): "))
    Sex = input("Sex (male or female): ").strip().lower()
    Age = float(input("Age: "))
    SibSp = int(input("Number of siblings/spouses aboard (SibSp): "))
    Parch = int(input("Number of parents/children aboard (Parch): "))
    Fare = float(input("Fare: "))
    Embarked = input("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()
    return {
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked': Embarked
    }

def main():
    # Load the fitted preprocessor
    preprocessor = joblib.load("fitted_preprocessor.joblib")
    
    # Get user input
    user_data = get_user_input()
    df = pd.DataFrame([user_data])
    
    # Preprocess input
    X_processed = preprocessor.transform(df)
    
    # Load the best XGBoost model
    model_path = Path("mlruns/211826327117354010/4350d73c13f64df9b743a4543441dcdf/artifacts/model")
    if not model_path.exists():
        print("Model not found. Please train the model first.")
        return
    model = mlflow.xgboost.load_model(str(model_path))
    
    # Predict
    pred = model.predict(X_processed)[0]
    proba = model.predict_proba(X_processed)[0, 1]
    print("\nPrediction result:")
    if pred == 1:
        print(f"This passenger would SURVIVE (probability: {proba:.2%})")
    else:
        print(f"This passenger would NOT survive (probability of survival: {proba:.2%})")

if __name__ == "__main__":
    main() 