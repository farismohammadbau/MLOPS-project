# Titanic Survival Prediction Project

This project uses machine learning to predict passenger survival on the Titanic based on various features like passenger class, age, gender, etc. The project is built using Python, scikit-learn, XGBoost, and MLflow for experiment tracking and model management.

## Project Structure

```
mlflow_project/
├── data/
│   ├── raw/
│   │   └── titanic.csv
│   └── processed/
├── models/
│   └── train_model.py
├── src/
│   ├── data/
│   │   └── process_data.py
│   ├── models/
│   │   └── train_model.py
│   ├── predict_terminal.py
│   └── serve_best_model.sh
└── requirements.txt
```

## Features

- Data preprocessing and feature engineering
- Multiple model training (Random Forest, Logistic Regression, XGBoost)
- Hyperparameter optimization using Hyperopt
- Model tracking and versioning with MLflow
- Terminal-based prediction interface
- Model serving capability

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd mlflow_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train the models:
```bash
cd src
python models/train_model.py
```

This will:
- Load and preprocess the Titanic dataset
- Train multiple models (Random Forest, Logistic Regression, XGBoost)
- Perform hyperparameter optimization
- Log experiments and models using MLflow

### Making Predictions

To make predictions using the terminal interface:
```bash
cd src
python predict_terminal.py
```

You'll be prompted to enter passenger details:
- Pclass (1, 2, or 3)
- Sex (male or female)
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Fare (typical range: 0-512)
- Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

### Serving the Model

To serve the best model:
```bash
cd src
bash serve_best_model.sh
```

The model will be served at `http://127.0.0.1:5001`

## Model Details

The project uses three different models:
1. Random Forest Classifier
2. Logistic Regression
3. XGBoost Classifier

Each model is trained with optimized hyperparameters using Hyperopt. The best performing model is automatically selected and can be used for predictions.

## Data Preprocessing

The preprocessing pipeline includes:
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Feature engineering

## Dependencies

- Python 3.8+
- scikit-learn
- XGBoost
- MLflow
- pandas
- numpy
- hyperopt

## License

[Your chosen license]

## Author

[Your name] 