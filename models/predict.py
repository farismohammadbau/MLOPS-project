import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
from data.process_data import load_data, create_preprocessing_pipeline

def load_model(run_id):
    """
    Load a model from MLflow using the run ID.
    """
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.xgboost.load_model(logged_model)
    return loaded_model

def make_predictions(model, data):
    """
    Make predictions using the loaded model.
    """
    predictions = model.predict(data)
    prediction_probas = model.predict_proba(data)
    return predictions, prediction_probas

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Load the raw data (with PassengerId)
    raw_df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
    passenger_ids = raw_df['PassengerId']
    df = load_data()  # This will drop PassengerId and preprocess
    
    # Preprocess the entire dataset
    X = df.drop('Survived', axis=1)
    preprocessor = create_preprocessing_pipeline()
    X_processed = preprocessor.fit_transform(X)
    
    # Get the latest run ID for the XGBoost model
    experiment = mlflow.get_experiment_by_name("titanic_survival")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    xgboost_runs = runs[runs['tags.mlflow.runName'] == 'xgboost_model']
    latest_run = xgboost_runs.iloc[xgboost_runs['start_time'].argmax()]
    run_id = latest_run['run_id']
    
    # Load the model
    print(f"Loading model from run ID: {run_id}")
    model = load_model(run_id)
    
    # Make predictions
    predictions, prediction_probas = make_predictions(model, X_processed)
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions,
        'Survival_Probability': prediction_probas[:, 1]
    })
    
    # Save predictions to CSV
    output_path = Path("predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    
    # Print some example predictions
    print("\nExample predictions:")
    print(results.head())
    
    # Print overall statistics
    print("\nPrediction Statistics:")
    print(f"Total passengers: {len(predictions)}")
    print(f"Predicted survivors: {sum(predictions)}")
    print(f"Predicted non-survivors: {len(predictions) - sum(predictions)}")
    print(f"Average survival probability: {np.mean(prediction_probas[:, 1]):.2%}")

if __name__ == "__main__":
    main() 