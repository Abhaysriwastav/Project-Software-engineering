# Stock Prediction Project

## Description

This project is a Django-based web application that provides stock price predictions using machine learning models. It incorporates historical stock data, allows user authentication, and presents predictions and visualizations in an accessible format. It leverages LSTM and Random Forest models for forecasting. The application includes comprehensive trend analysis and supports user-specific analysis and data updates. The front-end is designed with Bootstrap for responsiveness and Chart.js for interactive data visualization.

## Key Features

* **User Authentication:** User registration, login, password reset, and password change functionality powered by Django's built-in authentication.
* **Stock Price Prediction:** Predicts stock prices using pre-trained LSTM and Random Forest models, with dynamic model training if pre-trained models are unavailable.
* **Historical Data Display:** Shows historical stock data of top companies with interactive chart visualization, including customizable date ranges and key performance indicators.
* **Data Visualization:** Uses Chart.js for interactive visualizations of stock trends and prediction comparisons, enhancing user understanding of model outputs.
* **Data Handling:** Includes utilities for data preprocessing, synthetic data generation, and file management.
* **Technical Analysis:** Implements trend analysis algorithms, including moving averages, volatility calculation, and support/resistance identification.
* **Admin Interface:** Django admin interface for managing data and models.
* **Dynamic Data Updates:** Allows updating historical data through file uploads by staff users, ensuring the models stay current with market changes.
* **Robust Error Handling and Logging:** Extensive logging for debugging and monitoring, providing detailed insights into application behavior.

## Project Structure

```
stock_prediction/                # Root Project Directory
├── manage.py                   # Django management script
├── requirements.txt            # Project dependencies
├── db.sqlite3                  # SQLite database
├── data/                      # Project-level data storage
│   ├── top_companies_historical_data.csv
│   └── backups/              # Data backups directory
├── static/                    # Development static files
│   ├── css/
│   │   └── style.css
│   ├── images/
│   │   ├── background.jpg
│   │   └── logo.png
│   └── js/
│       ├── prediction.js
│       └── visualization.js
├── staticfiles/
│   └── admin/
├── Trained_models/
│   ├── lstm_model.pkl
│   └── rf_model.pkl
├── stock_app/                 # Main application
│   ├── templates/
│   ├── models/
│   ├── utils/
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── predictions.py
│   ├── urls.py
│   └── views.py
├── templates/                 # Project-level templates
│   └── base.html
└── stock_prediction/          # Project settings
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

## Installation

1. **Clone the repository:**
```bash
git clone  # if cloning from Git
cd stock_prediction
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required libraries:
```
Django==4.2.0
numpy==1.24.3
pandas==2.0.0
scikit-learn==1.2.2
tensorflow==2.12.0
matplotlib==3.7.1
seaborn==0.12.2
```

4. **Apply database migrations:**
```bash
python manage.py migrate
```

5. **Load historical data:**
* Ensure the `top_companies_historical_data.csv` file is present in the `data/` directory in both the project root and the `stock_app` directory.
* If the data needs to be loaded into the database, you can use a Django management command or a data loading script within the `stock_app` directory.
Note: make sure that data is preset in the correct directory..

6. **Runserver:**
```bash
python manage.py runserver
```
*It will open local host 
https://127.0.0.1:8000/predict/

## Methodology

The project employs a multi-faceted approach to stock price prediction, combining machine learning models with technical analysis techniques. Here's a breakdown of the methodology:

1. **Data Acquisition:** Historical stock data is acquired from a CSV file (`top_companies_historical_data.csv`).
2. **Data Preprocessing:** The data is preprocessed using the `DataPreprocessor` class, which handles missing values and removes outliers.
3. **Model Training:**
   * If pre-trained models (`lstm_model.pkl`, `rf_model.pkl`) are available, they are loaded.
   * Otherwise, the LSTM and Random Forest models are trained using the preprocessed historical data. The models are saved after training.
4. **Prediction:** The trained models are used to predict future stock prices based on the current price, market cap, and sector.
5. **Technical Analysis:** The `TrendAnalyzer` class is used to perform technical analysis, including calculating moving averages, detecting trends, and identifying support and resistance levels.
6. **Visualization:** The predicted prices and historical data are visualized using Chart.js.

## Core Components and Code Examples

Here are key components and code snippets that illustrate how the project works:

### Data Preprocessing

```python
# Example from stock_app/utils/data_handler.py
class DataPreprocessor:
    def __init__(self, data):
        self.data = data.copy()

    def handle_missing_values(self):
        for col in self.data.columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        return self.data

    def preprocess(self):
        self.data = self.handle_missing_values()
        return self.data
```

### Model Loading

```python
# Example from stock_app/predictions.py
def load_model(filename):
    try:
        filepath = os.path.join(settings.BASE_DIR, 'trained_models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model {filename}: {e}")
    return None
```

### Trend Analysis

```python
# Example from stock_app/utils/trend_analyzer.py
class TrendAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def calculate_moving_averages(self, company, windows=[5, 20, 50]):
        company_data = self.data[self.data['Company'] == company].sort_values('Date')

        results = {'company': company, 'moving_averages': {}}
        for window in windows:
            ma = company_data['Current Price (£)'].rolling(window=window).mean()
            results['moving_averages'][f'MA{window}'] = ma.iloc[-1]

        return results
```

## Models

### LSTM Model Implementation
```python
# Example code from stock_app/models/lstm_model.py
def build_model(self, input_shape):
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
```

### Random Forest Model Implementation
```python
# Example code from stock_app/models/random_forest_model.py
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)

    def prepare_data(self, data):
        features = ['Current Price (£)', 'Market Cap (£ m)']
        target = 'Current Price (£)'
        X = data[features].values
        y = data[target].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def train_and_evaluate(self, data):
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        metrics = {
            'mae': round(mean_absolute_error(y_test, y_pred), 4),
            'mse': round(mean_squared_error(y_test, y_pred), 4),
            'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'r2': round(r2_score(y_test, y_pred), 4),
            'mape': round(mean_absolute_percentage_error(y_test, y_pred), 4)
        }
        return metrics
```

## Dashboard Overview

1. **Stock Prediction Dashboard:**
   * **Input Section:** Allows users to input the company name, sector, market cap, and current price.
   ![Dashboard](Screenshots/Dashboard_page_1.png)
   ![Dashboard](Screenshots/Dashboard_page_1_half.png)
   * **Example Input:** You might input "Walmart Inc.", Sector "Retail", Market Cap "347.74", Current Price "460.24"
   * **Prediction Results:**
     * Displays predicted stock prices for the next 7 days using both models
     * Shows percentage change from current price
     * Displays model confidence level
     * Presents performance metrics (MAE, RMSE, R2)
   * **Example Predictions:** 
     - LSTM Prediction: €427.13 (Change: -7.20%)
     - Random Forest Prediction: €445.42 (Change: -3.22%)
    ![Dashboard](Screenshots/Dashboard_Generating_prediction_loading_page.png) 
    ![Dashboard](Screenshots/Dashboard_predection_result_page_1.png)
    ![Dashboard](Screenshots/Dashboard_predection_result_page_2.png)    
        


2. **Historical Data Dashboard:**
   * Allows users to select a company and date range
   * Displays key performance indicators
   * Shows stock price history chart
   * Presents trend analysis information
![Historical](Screenshots/Historical_data_result_page_1.png)
![Historical](Screenshots/Historical_data_result_page_2.png)
![Historical](Screenshots/Historical_data_result_page_3.png)

## Dataset Example
```
| Date       | Company           | Sector     | Price | Market Cap | Volume  |
|------------|------------------|------------|-------|------------|---------|
| 03-02-2020 | UnitedHealth     | Healthcare | 451.74| 915205.66  | 35656907|
```

## Future Scope

* **Integration of Real-Time Data:** Connect to a real-time stock data API
* **Advanced Technical Indicators:** Add MACD, RSI, Fibonacci retracements
* **Sentiment Analysis:** Integrate news and social media analysis
* **Portfolio Management:** Add virtual portfolio features
* **User Customization:** Allow model and analysis customization
* **Enhanced Visualization:** Improve interactive charts and graphs
* **Deployment:** Deploy to production environment
* **Model Improvements:** Add hyperparameters and algorithm updates

## Screenshots
![Login](Screenshots/Login_Page.png)
![Logout](Screenshots/Logout_page.png)
![Profile](Screenshots/Profile.png)
![reset](Screenshots/Reset_password_page.png)
![Register](Screenshots/Register_Page.png)



## Author
Abhay sriwastav
