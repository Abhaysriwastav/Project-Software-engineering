# Stock Market Prediction Platform

## 🚀 Project Overview

A sophisticated Django-based Stock Market Prediction Platform that leverages advanced machine learning techniques to forecast stock prices using LSTM and Random Forest models.

## 📊 Project Relevance and Importance

### Why Stock Market Prediction Matters
- **Financial Decision Making**: Provides data-driven insights for investors
- **Risk Management**: Helps in understanding potential stock price movements
- **Machine Learning Application**: Demonstrates complex ML models in financial domain
- **Technological Innovation**: Combines web technologies with predictive analytics

## 🔍 Key Features

- **User Authentication**: Secure login and registration system
- **Advanced Prediction Models**:
  - Long Short-Term Memory (LSTM) Neural Network
  - Random Forest Regression
- **Comprehensive Data Visualization**
- **Historical Data Analysis**
- **Trend Detection**

## 💻 Technology Stack

### Backend
- Django 4.2.0
- Python 3.x

### Machine Learning
- TensorFlow
- Scikit-learn
- NumPy
- Pandas

### Frontend
- Bootstrap 5
- Chart.js
- Font Awesome

### Database
- SQLite (Development)
- Redis Caching

## 🧠 Machine Learning Methodology

### Data Preprocessing
- Missing value handling
- Outlier removal
- Feature scaling
- Data normalization

### Model Architectures
1. **LSTM Model**
   - Two-layer LSTM architecture
   - BatchNormalization
   - Dropout for regularization
   - Early stopping

2. **Random Forest Model**
   - 200 estimators
   - Advanced hyperparameter tuning
   - Feature importance analysis

### Performance Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## 📁 Project Structure

```
stock_prediction/
│
├── manage.py
├── requirements.txt
│
├── data/
│   └── top_companies_historical_data.csv
│
├── stock_app/
│   ├── models/
│   │   ├── lstm_model.py
│   │   └── random_forest_model.py
│   │
│   ├── utils/
│   │   ├── data_handler.py
│   │   ├── trend_analyzer.py
│   │   └── visualizer.py
│   │
│   └── views.py
│
└── stock_prediction/
    └── settings.py
```

## 🖥️ User Interface Screenshots

### Login Page
![Login Page](/screenshots/login.png)

### Prediction Dashboard
![Prediction Dashboard](/screenshots/prediction_dashboard.png)

### Historical Data Analysis
![Historical Data](/screenshots/historical_data.png)

## 📈 Dataset Details

### Data Source
- **Filename**: `top_companies_historical_data.csv`
- **Total Rows**: 40,150
- **Columns**:
  1. Date
  2. Company
  3. Sector
  4. Country
  5. Current Price (£)
  6. Market Cap (£ m)
  7. Trading Volume
  8. Open Price (£)
  9. High Price (£)
  10. Low Price (£)
  11. Price Change (%)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip
- Virtual Environment

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stock-prediction-platform.git
cd stock-prediction-platform
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run migrations
```bash
python manage.py migrate
```

5. Start the development server
```bash
python manage.py runserver
```

## 🔬 Model Strengths

1. **Robust Preprocessing**
   - Handles missing and infinite values
   - Advanced feature scaling

2. **Multiple Prediction Models**
   - LSTM for sequence learning
   - Random Forest for ensemble prediction

3. **Comprehensive Evaluation**
   - Multiple performance metrics
   - Cross-validation

4. **Scalable Architecture**
   - Modular design
   - Easy model extension

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - youremail@example.com

Project Link: [https://github.com/yourusername/stock-prediction-platform](https://github.com/yourusername/stock-prediction-platform)
