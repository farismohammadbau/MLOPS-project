import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from pathlib import Path
import os

def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load and preprocess the Titanic dataset.
    
    Args:
        data_path (str): Path to the dataset file. If None, downloads from URL.
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    if data_path is None:
        # Download the dataset if not provided
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(data_path)
    
    # Basic preprocessing
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    return df

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the Titanic dataset.
    
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataset
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Fit and transform the training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    # Set up MLflow experiment
    mlflow.set_experiment("titanic_survival")
    
    with mlflow.start_run():
        # Load and process data
        df = load_data()
        
        # Log dataset information
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("features", list(df.columns))
        
        # Split data
        X_train, X_test, y_train, y_test, preprocessor = split_data(df)
        
        # Log data split information
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        print("Data processing completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}") 