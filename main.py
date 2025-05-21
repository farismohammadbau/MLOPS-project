import mlflow
from pathlib import Path
import os
import time
from data.process_data import load_data, split_data
from models.train_model import train_model, evaluate_model, optimize_hyperparameters
from deployment.deploy_model import deploy_model, register_model, transition_model
from monitoring.monitor_model import ModelMonitor

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5002")
    
    # Experiment 1: Original Configuration
    experiment = mlflow.get_experiment_by_name("titanic_survival_original")
    if experiment is None:
        experiment_id = mlflow.create_experiment("titanic_survival_original")
        mlflow.set_experiment(experiment_id)
    else:
        mlflow.set_experiment(experiment.experiment_id)
    
    with mlflow.start_run():
        # 1. Data Processing
        print("Step 1: Processing data...")
        df = load_data()
        print(f"Loaded data shape: {df.shape}")
        print(f"Features: {df.columns.tolist()}")
        print(f"Target distribution:\n{df['Survived'].value_counts()}")
        
        X_train, X_test, y_train, y_test, preprocessor = split_data(df)
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Log data information
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", df.columns.tolist())
        
        # 2. Model Training and Optimization
        print("\nStep 2: Training and optimizing model...")
        best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
        print(f"Best parameters: {best_params}")
        
        final_model = train_model(X_train, y_train, best_params)
        final_metrics = evaluate_model(final_model, X_test, y_test)
        print(f"Final metrics: {final_metrics}")
        
        # Log model information
        mlflow.log_params(best_params)
        mlflow.log_metrics(final_metrics)
        model_uri = mlflow.sklearn.log_model(final_model, "model").model_uri
        print(f"Model URI: {model_uri}")
        
        # 3. Model Deployment
        print("\nStep 3: Deploying model...")
        model_name = "titanic_survival_model_original"
        process = deploy_model(model_uri)
        
        if process:
            try:
                # Register and transition model
                register_model(model_uri, model_name)
                transition_model(model_name, 1, "Staging")
                
                # 4. Model Monitoring
                print("\nStep 4: Setting up model monitoring...")
                monitor = ModelMonitor(model_name, 1)
                
                # Example predictions and monitoring
                example_data = {
                    "dataframe_split": {
                        "columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
                        "data": [[3, "male", 22.0, 1, 0, 7.25, "S"]]
                    }
                }
                
                print("\nMaking example predictions:")
                for i in range(5):
                    prediction = monitor.make_prediction(example_data)
                    print(f"Prediction {i+1}:", prediction)
                    time.sleep(1)  # Simulate real-time predictions
                
                # Generate monitoring report
                report = monitor.generate_report()
                print("\nMonitoring Report:")
                print(report)
                
                # Keep the server running
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                print("\nModel serving stopped")
        else:
            print("Failed to deploy model")

    # Experiment 2: Feature Engineering Focus
    experiment = mlflow.get_experiment_by_name("titanic_survival_feature_engineering")
    if experiment is None:
        experiment_id = mlflow.create_experiment("titanic_survival_feature_engineering")
        mlflow.set_experiment(experiment_id)
    else:
        mlflow.set_experiment(experiment.experiment_id)
    
    with mlflow.start_run():
        print("\nStarting Feature Engineering Experiment...")
        # Add feature engineering steps here
        df = load_data()
        print(f"Original data shape: {df.shape}")
        
        # Add family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        # Add fare per person
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        
        print(f"Enhanced data shape: {df.shape}")
        print(f"New features: {['FamilySize', 'FarePerPerson']}")
        
        X_train, X_test, y_train, y_test, preprocessor = split_data(df)
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Log data information
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("additional_features", ["FamilySize", "FarePerPerson"])
        
        # Model Training and Optimization
        best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
        print(f"Best parameters: {best_params}")
        
        final_model = train_model(X_train, y_train, best_params)
        final_metrics = evaluate_model(final_model, X_test, y_test)
        print(f"Final metrics: {final_metrics}")
        
        # Log model information
        mlflow.log_params(best_params)
        mlflow.log_metrics(final_metrics)
        model_uri = mlflow.sklearn.log_model(final_model, "model").model_uri
        print(f"Model URI: {model_uri}")
        
        # Model Deployment
        model_name = "titanic_survival_model_feature_engineering"
        process = deploy_model(model_uri)
        
        if process:
            try:
                register_model(model_uri, model_name)
                transition_model(model_name, 1, "Staging")
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                print("\nModel serving stopped")
        else:
            print("Failed to deploy model")

    # Experiment 3: Hyperparameter Tuning Focus
    experiment = mlflow.get_experiment_by_name("titanic_survival_hyperparameter_tuning")
    if experiment is None:
        experiment_id = mlflow.create_experiment("titanic_survival_hyperparameter_tuning")
        mlflow.set_experiment(experiment_id)
    else:
        mlflow.set_experiment(experiment.experiment_id)
    
    with mlflow.start_run():
        print("\nStarting Hyperparameter Tuning Experiment...")
        df = load_data()
        print(f"Data shape: {df.shape}")
        
        X_train, X_test, y_train, y_test, preprocessor = split_data(df)
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Log data information
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Extended hyperparameter search space
        best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test, max_evals=100)
        print(f"Best parameters: {best_params}")
        
        final_model = train_model(X_train, y_train, best_params)
        final_metrics = evaluate_model(final_model, X_test, y_test)
        print(f"Final metrics: {final_metrics}")
        
        # Log model information
        mlflow.log_params(best_params)
        mlflow.log_metrics(final_metrics)
        model_uri = mlflow.sklearn.log_model(final_model, "model").model_uri
        print(f"Model URI: {model_uri}")
        
        # Model Deployment
        model_name = "titanic_survival_model_hyperparameter_tuning"
        process = deploy_model(model_uri)
        
        if process:
            try:
                register_model(model_uri, model_name)
                transition_model(model_name, 1, "Staging")
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                print("\nModel serving stopped")
        else:
            print("Failed to deploy model")

if __name__ == "__main__":
    main() 