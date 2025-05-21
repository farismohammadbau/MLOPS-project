import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from data.process_data import create_preprocessing_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_serving.log'),
        logging.StreamHandler()
    ]
)

def serve_model():
    """Serve the model using MLflow's serving capabilities"""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Load the model
        model_path = Path("mlruns/211826327117354010/4350d73c13f64df9b743a4543441dcdf/artifacts/model")
        if not model_path.exists():
            logging.error("Model not found. Please train the model first.")
            return
            
        # Serve the model
        logging.info("Starting model serving...")
        mlflow.xgboost.serve_model(
            model_uri=str(model_path),
            host="127.0.0.1",
            port=5000,
            enable_mlserver=True
        )
        
    except Exception as e:
        logging.error(f"Error serving model: {str(e)}")

if __name__ == "__main__":
    serve_model() 