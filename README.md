# Titanic Survival Prediction Model - Deployment and Monitoring

This project implements a machine learning model for predicting Titanic passenger survival, with deployment and monitoring capabilities using MLflow.

## Model Deployment

The model is deployed using MLflow's model serving capabilities. To start the model server:

```bash
python serve_model.py
```

This will start the model server on `http://127.0.0.1:5000`.

## Model Monitoring

The monitoring system tracks the following metrics:
- Total number of predictions
- Survival rate (percentage of predicted survivors)
- Average confidence score
- Timestamp of predictions

### Monitoring Files
- `model_monitoring.log`: Contains detailed logs of all predictions and metrics
- `prediction_logs.json`: Stores the history of predictions with input data and results

### Running the Monitoring System

To simulate incoming data and monitor model performance:

```bash
python deploy_model.py
```

This will:
1. Load the trained model
2. Simulate 10 random passenger predictions
3. Log predictions and calculate performance metrics
4. Save results to monitoring files

## Performance Metrics

The monitoring system calculates the following metrics over a sliding window of 100 predictions:
- `total_predictions`: Number of predictions in the window
- `survival_rate`: Percentage of passengers predicted to survive
- `avg_confidence`: Average confidence score of predictions
- `timestamp`: When the metrics were calculated

## Logging

All activities are logged to:
- `model_monitoring.log`: For prediction and monitoring activities
- `model_serving.log`: For model serving activities

## API Endpoints

When the model is served, it exposes the following endpoints:
- `POST /invocations`: Make predictions
- `GET /ping`: Health check

Example prediction request:
```bash
curl -X POST http://127.0.0.1:5000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": 1,
      "Sex": "female",
      "Age": 30,
      "SibSp": 0,
      "Parch": 0,
      "Fare": 50,
      "Embarked": "C"
    }
  }'
```

## Monitoring Dashboard

To view the monitoring dashboard:
1. Start the MLflow UI:
```bash
mlflow ui
```
2. Open `http://localhost:5000` in your browser
3. Navigate to the "titanic_survival" experiment
4. View the metrics and model performance over time 