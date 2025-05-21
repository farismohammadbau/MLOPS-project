import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from data.process_data import load_data, split_data
import joblib

def train_model(X_train, y_train, params, model_type='random_forest'):
    """
    Train a model with the given parameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Dictionary of hyperparameters
        model_type: Type of model to train ('random_forest', 'logistic', 'xgboost')
        
    Returns:
        Trained model
    """
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=params['C'],
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using various metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

def optimize_hyperparameters(X_train, y_train, X_test, y_test, model_type='random_forest'):
    """
    Optimize hyperparameters using Hyperopt.
    """
    if model_type == 'random_forest':
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 50),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5)
        }
    elif model_type == 'logistic':
        space = {
            'C': hp.loguniform('C', np.log(0.001), np.log(100))
        }
    elif model_type == 'xgboost':
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 50),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
        }

    def objective(params):
        model = train_model(X_train, y_train, params, model_type)
        metrics = evaluate_model(model, X_test, y_test)
        return {'loss': -metrics['roc_auc'], 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)
    
    return best

if __name__ == "__main__":
    # Set up MLflow experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("titanic_survival")
    
    # Load and process data
    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = split_data(df)
    
    # Save the fitted preprocessor
    joblib.dump(preprocessor, "fitted_preprocessor.joblib")
    
    # Train and evaluate different models
    models = ['random_forest', 'logistic', 'xgboost']
    
    for model_type in models:
        with mlflow.start_run(run_name=f"{model_type}_model"):
            print(f"\nTraining {model_type} model...")
            
            # Optimize hyperparameters
            best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, model_type)
            print(f"Best parameters for {model_type}:", best_params)
            
            # Train final model with best parameters
            final_model = train_model(X_train, y_train, best_params, model_type)
            final_metrics = evaluate_model(final_model, X_test, y_test)
            print(f"Final metrics for {model_type}:", final_metrics)
            
            # Log parameters and metrics
            mlflow.log_params(best_params)
            mlflow.log_metrics(final_metrics)
            
            # Log model
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(final_model, "model")
            else:
                mlflow.sklearn.log_model(final_model, "model")
            
            print(f"{model_type} model training completed successfully!") 