import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMModel:
    def __init__(self):
        self.model = None
        # self.scaler = MinMaxScaler()
        self.scaler_X = MinMaxScaler()  # Scaler for features
        self.scaler_y = MinMaxScaler()
        self.metrics = {}
    
    # def prepare_data(self, data):
    #     """
    #     Prepare data for LSTM model with robust preprocessing
    #     """
    #     try:
    #         # Select features for prediction
    #         features = ['Current Price (£)', 'Market Cap (£ m)']
    #         target = 'Current Price (£)'
            
    #         # Extract features and target
    #         X = data[features].values
    #         y = data[target].values
            
    #         # Handle potential problematic values
    #         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    #         y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
    #         # Remove rows with infinite values
    #         mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    #         X = X[mask]
    #         y = y[mask]
            
    #         # Scale the data
    #         X_scaled = self.scaler.fit_transform(X)
    #         y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
            
    #         return X_scaled, y_scaled
        
    #     except Exception as e:
    #         print(f"Data preparation error: {e}")
    #         return None, None
    def prepare_data(self, data):
        """
        Prepare data for LSTM model with robust preprocessing
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
            
            # Remove rows with infinite values
            mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[mask]
            y = y[mask]
            
            # Scale the data using the single scaler
            X_scaled = np.zeros_like(X)
            for i in range(X.shape[1]):
                feature = X[:, i].reshape(-1, 1)
                X_scaled[:, i] = self.scaler.fit_transform(feature).flatten()
            
            y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
            
            return X_scaled, y_scaled
        
        except Exception as e:
            print(f"Data preparation error: {e}")
            return None, None
    
    def build_model(self, input_shape):
        """Build advanced LSTM model architecture"""
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.LSTM(32)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train_and_evaluate(self, data):
        """Comprehensive model training and evaluation"""
        try:
            # Prepare data
            X, y = self.prepare_data(data)
            
            if X is None or y is None:
                print("Data preparation failed")
                return {}
            
            # Reshape data for LSTM
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build and train model
            self.model = self.build_model((1, X.shape[2]))
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
            
            self.model.fit(
                X_train, y_train, 
                epochs=100, 
                batch_size=32, 
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Predict and calculate metrics
            y_pred = self.model.predict(X_test, verbose=0)
            y_pred_inv = self.scaler.inverse_transform(y_pred)
            y_test_inv = self.scaler.inverse_transform(y_test)
            
            # Calculate performance metrics
            self.metrics = {
                'mae': float(mean_absolute_error(y_test_inv, y_pred_inv)),
                'mse': float(mean_squared_error(y_test_inv, y_pred_inv)),
                'rmse': float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
                'r2': float(r2_score(y_test_inv, y_pred_inv))
            }
            
            return self.metrics
        
        except Exception as e:
            print(f"Training error: {e}")
            return {}
    
    # def predict(self, input_data):
    #     """Robust prediction method"""
    #     try:
    #         # Ensure input is a numpy array
    #         input_data = np.array(input_data).reshape(1, -1)
    #         # print("input_data : ",input_data)
    #         # print("="*50)
    #         # Scale input data
    #         # input_scaled = self.scaler.transform(input_data)
    #         input_scaled = self.scaler.fit_transform(input_data)
    #         input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    #         print(input_reshaped)
    #         print("="*50)
    #         # Make prediction
    #         prediction = self.model.predict(input_reshaped, verbose=0)
    #         print(prediction.shape)
    #         print(prediction)
    #         print("="*50)
    #         # Inverse transform prediction
    #         prediction_inv = self.scaler.inverse_transform(prediction)
            
    #         return prediction_inv.flatten()
    #         # return prediction
        
    #     except Exception as e:
    #         print(f"Prediction error: {e}")
    #         raise

    def predict(self, input_data):
        """Robust prediction method"""
        try:
            # Ensure input is a numpy array
            input_data = np.array(input_data).reshape(1, -1)
            
            # Scale each feature separately
            input_scaled = np.zeros_like(input_data)
            for i in range(input_data.shape[1]):
                feature = input_data[:, i].reshape(-1, 1)
                input_scaled[:, i] = self.scaler.transform(feature).flatten()
            
            # Reshape for LSTM input [samples, timesteps, features]
            input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
            
            # Make prediction
            prediction = self.model.predict(input_reshaped, verbose=0)
            
            # Inverse transform the prediction
            prediction_inv = self.scaler.inverse_transform(prediction)
            
            return prediction_inv.flatten()
            
        except Exception as e:
            print(f"Prediction error: {e}")
            raise