import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import yaml
import time
import os
from datetime import datetime

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_model():
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    model_uri = f"models:/{config['model']['name']}/{config['model']['stage']}"
    return mlflow.sklearn.load_model(model_uri)

def calculate_drift(current_metrics, baseline_metrics, threshold):
    drift_detected = False
    drift_metrics = {}
    
    for metric in current_metrics:
        if metric in baseline_metrics:
            diff = abs(current_metrics[metric] - baseline_metrics[metric])
            if diff > threshold:
                drift_detected = True
                drift_metrics[metric] = {
                    'current': current_metrics[metric],
                    'baseline': baseline_metrics[metric],
                    'difference': diff
                }
    
    return drift_detected, drift_metrics

def monitor_model():
    config = load_config()
    model = load_model()
    
    # Load baseline metrics from MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    experiment = mlflow.get_experiment_by_name(config['mlflow']['experiment_name'])
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    baseline_run = runs.iloc[0]
    baseline_metrics = {
        'accuracy': baseline_run['metrics.accuracy'],
        'precision': baseline_run['metrics.precision'],
        'recall': baseline_run['metrics.recall'],
        'f1': baseline_run['metrics.f1']
    }
    
    while True:
        try:
            # Simulate new data (in production, this would be real data)
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = pd.Series(iris.target, name='target')
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate current metrics
            current_metrics = {
                'accuracy': accuracy_score(y, predictions)
            }
            
            # Check for drift
            drift_detected, drift_metrics = calculate_drift(
                current_metrics,
                baseline_metrics,
                config['monitoring']['drift_threshold']
            )
            
            # Log monitoring results
            with mlflow.start_run(nested=True):
                mlflow.log_metrics(current_metrics)
                mlflow.log_param('drift_detected', drift_detected)
                if drift_detected:
                    mlflow.log_dict(drift_metrics, 'drift_metrics.json')
                    print(f"Drift detected at {datetime.now()}")
                    print(f"Drift metrics: {drift_metrics}")
            
            # Wait for next monitoring interval
            time.sleep(config['monitoring']['check_interval'])
            
        except Exception as e:
            print(f"Error in monitoring: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == '__main__':
    monitor_model() 