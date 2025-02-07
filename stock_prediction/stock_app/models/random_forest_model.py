import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)

# Configure logging
logger = logging.getLogger(__name__)

class RandomForestModel:
    def __init__(self):
        """
        Initialize Random Forest model with enhanced parameters
        """
        self.model = RandomForestRegressor(
            n_estimators=200,  # Increased number of estimators
            max_depth=None,  # Allow full depth
            min_samples_split=5,  # Minimum samples to split an internal node
            min_samples_leaf=2,  # Minimum samples in a leaf node
            max_features='sqrt',  # Number of features to consider for best split
            bootstrap=True,  # Use bootstrap samples
            n_jobs=-1,  # Use all available cores
            random_state=42  # Ensure reproducibility
        )
        self.scaler = StandardScaler()
    
    def prepare_data(self, data):
        """
        Comprehensive data preparation for Random Forest model
        
        Args:
            data (pandas.DataFrame): Input data
        
        Returns:
            tuple: Prepared X and y data
        """
        try:
            # Select features for prediction
            features = ['Current Price (£)', 'Market Cap (£ m)']
            target = 'Current Price (£)'
            
            # Extract features and target
            X = data[features].values
            y = data[target].values
            
            # Handle potential problematic values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
        
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            return None, None
    
    def train_and_evaluate(self, data):
        """
        Comprehensive model training and evaluation
        
        Args:
            data (pandas.DataFrame): Training data
        
        Returns:
            dict: Detailed model performance metrics
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            # Validate data preparation
            if X is None or y is None:
                logger.error("Data preparation failed")
                return self._get_default_metrics()
            
            # Validate data dimensions
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid data for training")
                return self._get_default_metrics()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                'mae': round(mean_absolute_error(y_test, y_pred), 4),
                'mse': round(mean_squared_error(y_test, y_pred), 4),
                'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                'r2': round(r2_score(y_test, y_pred), 4),
                'mape': round(mean_absolute_percentage_error(y_test, y_pred), 4)
            }
            
            # Optional: Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_squared_error')
            metrics['cv_rmse'] = round(np.sqrt(-cv_scores.mean()), 4)
            
            # Feature importance
            feature_importance = self.model.feature_importances_
            metrics['feature_importance'] = feature_importance.tolist()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error in Random Forest model training: {e}", exc_info=True)
            return self._get_default_metrics()
    
    def predict(self, input_data):
        """
        Robust prediction method
        
        Args:
            input_data (list/numpy.ndarray): Input features for prediction
        
        Returns:
            numpy.ndarray: Predicted values
        """
        try:
            # Ensure input is a numpy array
            input_data = np.array(input_data)
            
            # Handle potential shape issues
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)
            
            # Handle potential problematic values
            input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}", exc_info=True)
            # Return original input as fallback
            return np.array(input_data)
    
    def _get_default_metrics(self):
        """
        Generate default metrics when training fails
        
        Returns:
            dict: Default performance metrics
        """
        return {
            'mae': 0,
            'mse': 0,
            'rmse': 0,
            'r2': 0,
            'mape': 0,
            'cv_rmse': 0,
            'feature_importance': []
        }