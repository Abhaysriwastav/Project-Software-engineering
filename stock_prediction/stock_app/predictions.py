import numpy as np
import logging
from django.conf import settings
import os
import pickle

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

def predict_stock(company_name, sector, market_cap, current_price, additional_training_data=None):
    """
    Comprehensive stock price prediction function
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
            stored_lstm_metrics = {}
            stored_rf_metrics = {}
        
        # Validate input features
        if len(input_features) != 2:
            raise ValueError("Input features must be [current_price, market_cap]")
        
        print(input_features)
        print("=================================")
        
        # Make predictions
        lstm_prediction = lstm_model.predict(input_features)[0]
        rf_prediction = rf_model.predict(input_features)[0]
        
        # Debug logging
        logger.info(f"Predictions for {company_name}: LSTM={lstm_prediction}, RF={rf_prediction}")
        
        return {
                    'predictions': {
                        'lstm_prediction': float(lstm_prediction),
                        'rf_prediction': float(rf_prediction)
                    },
                    # This is causing the error because it's trying to retrain the model:
                    'lstm_metrics': lstm_model.train_and_evaluate(additional_training_data) if additional_training_data is not None else {},
                    'rf_metrics': rf_model.train_and_evaluate(additional_training_data) if additional_training_data is not None else {}
                }
    
    except Exception as e:
        # Log the full error
        logger.error(f"Prediction error for {company_name}: {e}", exc_info=True)
        
        # Return fallback prediction
        return {
            'predictions': {
                'lstm_prediction': current_price,
                'rf_prediction': current_price
            },
            'error': str(e)
        }