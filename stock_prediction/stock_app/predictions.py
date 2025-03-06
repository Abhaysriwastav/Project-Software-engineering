import numpy as np
import logging
from django.conf import settings
import os
import pickle
import pandas as pd

# Import your models
from .models.lstm_model import LSTMModel
from .models.random_forest_model import RandomForestModel

# Configure logging
logger = logging.getLogger(__name__)

def save_model(model, filename):
    """Save a trained model to disk"""
    try:
        os.makedirs(os.path.join(settings.BASE_DIR, 'trained_models'), exist_ok=True)
        filepath = os.path.join(settings.BASE_DIR, 'trained_models', filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        logger.error(f"Error saving model {filename}: {e}")

def load_model(filename):
    """Load a trained model from disk"""
    try:
        filepath = os.path.join(settings.BASE_DIR, 'trained_models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model {filename}: {e}")
    return None

def predict_stock(company_name, sector, market_cap, current_price, additional_training_data=None, num_days=7):
    """
    Comprehensive stock price prediction function for multiple days
    """
    try:
        # Prepare input features
        input_features = [
            current_price,     # Current Price
            market_cap         # Market Cap
        ]

        # Attempt to load pre-trained models
        lstm_model = load_model('lstm_model.pkl')
        rf_model = load_model('rf_model.pkl')

        # If models are not loaded, train new models
        if lstm_model is None or rf_model is None:
            if additional_training_data is None or additional_training_data.empty:
                raise ValueError("No training data available to train models")

            # Train LSTM Model
            lstm_model = LSTMModel()
            lstm_metrics = lstm_model.train_and_evaluate(additional_training_data)
            save_model(lstm_model, 'lstm_model.pkl')

            # Train Random Forest Model
            rf_model = RandomForestModel()
            rf_metrics = rf_model.train_and_evaluate(additional_training_data)
            save_model(rf_model, 'rf_model.pkl')

            # Store metrics from initial training
            stored_lstm_metrics = lstm_metrics
            stored_rf_metrics = rf_metrics
        else:
            # Use empty metrics if using pre-trained models
            # Evaluate the models on a small sample of the training data
            if additional_training_data is not None and not additional_training_data.empty:
                # Take the first 100 rows as a sample
                sample_data = additional_training_data.head(100)
                lstm_metrics = lstm_model.train_and_evaluate(sample_data)
                rf_metrics = rf_model.train_and_evaluate(sample_data)

                stored_lstm_metrics = lstm_metrics
                stored_rf_metrics = rf_metrics
            else:
                stored_lstm_metrics = {}
                stored_rf_metrics = {}


        # Validate input features
        if len(input_features) != 2:
            raise ValueError("Input features must be [current_price, market_cap]")

        print(input_features)
        print("=================================")

        lstm_predictions = []
        rf_predictions = []
        current_lstm_price = current_price
        current_rf_price = current_price

        for _ in range(num_days):
            # LSTM Prediction
            lstm_input = [current_lstm_price, market_cap]
            lstm_prediction = lstm_model.predict(lstm_input)[0]
            lstm_predictions.append(float(lstm_prediction))
            current_lstm_price = lstm_prediction  # Update for next iteration

            # Random Forest Prediction
            rf_input = [current_rf_price, market_cap]
            rf_prediction = rf_model.predict(rf_input)[0]
            rf_predictions.append(float(rf_prediction))
            current_rf_price = rf_prediction  # Update for next iteration
            
        # Debug logging
        logger.info(f"Predictions for {company_name}: LSTM={lstm_predictions}, RF={rf_predictions}")

        logger.info(f"LSTM Metrics: {stored_lstm_metrics}")
        logger.info(f"RF Metrics: {stored_rf_metrics}")

        return {
            'predictions': {
                'lstm_predictions': lstm_predictions,
                'rf_predictions': rf_predictions
            },
            'lstm_metrics': stored_lstm_metrics,  # Use metrics from initial training or empty dict
            'rf_metrics': stored_rf_metrics  # Use metrics from initial training or empty dict
        }

    except Exception as e:
        # Log the full error
        logger.error(f"Prediction error for {company_name}: {e}", exc_info=True)

        # Return fallback prediction
        return {
            'predictions': {
                'lstm_predictions': [current_price] * num_days,
                'rf_predictions': [current_price] * num_days
            },
            'error': str(e)
        }