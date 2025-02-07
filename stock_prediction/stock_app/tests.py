from django.test import TestCase
from .models import LSTMModel, RandomForestModel
import numpy as np

class ModelTests(TestCase):
    def setUp(self):
        self.lstm_model = LSTMModel()
        self.rf_model = RandomForestModel()
    
    def test_lstm_predictions(self):
        # Create sample data
        X = np.random.random((100, 60, 1))
        y = np.random.random(100)
        
        # Test training
        self.lstm_model.train(X, y)
        
        # Test predictions
        predictions = self.lstm_model.predict(X)
        self.assertEqual(len(predictions), len(X))