import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from data.process_data import create_preprocessing_pipeline

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
    # Get user input
    user_data = get_user_input()
    df = pd.DataFrame([user_data])
    
    # Preprocess input
    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(df)
    
    # Load the model directly from the mlruns directory
    model_path = Path("mlruns/1/latest/artifacts/model")
    if not model_path.exists():
        print("Model not found. Please train the model first.")
        return
        
    try:
        model = mlflow.xgboost.load_model(str(model_path))
        
        # Predict
        pred = model.predict(X_processed)[0]
        proba = model.predict_proba(X_processed)[0, 1]
        print("\nPrediction result:")
        if pred == 1:
            print(f"This passenger would SURVIVE (probability: {proba:.2%})")
        else:
            print(f"This passenger would NOT survive (probability of survival: {proba:.2%})")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure you have trained the model first.")

if __name__ == "__main__":
    main() 