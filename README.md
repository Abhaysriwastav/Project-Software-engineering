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
git clone [your_repository_url]
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

The project requires the following libraries:
```
Django==4.2.0
numpy==1.24.3
pandas==2.0.0
scikit-learn==1.2.2
tensorflow==2.12.0
matplotlib==3.7.1
seaborn==0.12.2
```

4. **Configure environment variables:**
* Set up your environment variables according to the project documentation.
* Ensure all required configuration variables are properly set for development and production environments.

5. **Apply database migrations:**
```bash
python manage.py migrate
```

6. **Load historical data:**
* Ensure the `top_companies_historical_data.csv` file is present in the `data/` directory in both the project root and the `stock_app` directory.
* If the data needs to be loaded into the database, you can use a Django management command or a data loading script within the `stock_app` directory.

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

## Dashboard Overview

The application provides two main dashboards:

1. **Stock Prediction Dashboard:**
   * **Input Section:** Allows users to input the company name, sector, market cap, and current price.
   * **Screenshot of Input Parameters**
     ![Stock Data Input](link_to_you![Dashboard_predection_result_page_1](https://github.com/user-attachments/assets/aef08383-30b7-4a03-8024-59d3474ec44f)
r_stock_data_input_screenshot.png)
     
   * **Prediction Results:**
     * Displays the predicted stock prices for the next 7 days using both the LSTM and Random Forest models.
     * Shows the percentage change from the current price.
     * Displays the model's confidence level.
     * Presents performance metrics for each model (MAE, RMSE, R2).
     * Includes a chart comparing the predictions of the two models over the 7-day period.
     * Features a chart comparing the performance metrics of the two models.
     ![Stock Prediction Dashboard](link_to_your_stock_prediction_screenshot.png)
   * **Dataset Example (top_companies_historical_data.csv):**
     | Date       | Company           | Sector     | Country | Current Price | Market Cap | Trading Volume | Open Price | High Price | Low Price | Price Change (%) |
     |------------|-------------------|------------|---------|---------------|------------|----------------|------------|------------|-----------|------------------|
     | 03-02-2020 | UnitedHealth Group| Healthcare | USA     | 451.74        | 915205.66  | 35656907       | 456.12     | 460.94     | 449.25    | 0.39             |

2. **Historical Data Dashboard:**
   * Allows users to select a company and a date range to view historical stock data.
     ![Historical Dashboard Parameters](link_to_your_historical_data_dashboard_parameters_selection.png)
   * Displays key performance indicators (trading days, low price, high price, average volume).
   * Presents a chart of the stock price history over the selected period.
   * Displays trend analysis information, including moving averages, trend direction and strength, and support and resistance levels.
     ![Historical Data Dashboard](link_to_your_historical_data_screenshot.png)

## Models

* **LSTM Model (`lstm_model.py`):** A Long Short-Term Memory neural network model implemented using TensorFlow. Used for time series forecasting of stock prices. The LSTM model is constructed with the following:
  * **Input:** The model takes sequences of stock data as input.
  * **Layers:** It consists of LSTM layers, Batch Normalization, and Dropout to prevent overfitting.
  * **Output:** Produces a prediction of the next stock price.
  * **Training:** Adam optimizer is used to minimize the mean squared error between predicted and actual prices.

* **Random Forest Model (`random_forest_model.py`):** An ensemble learning method based on decision trees, implemented using scikit-learn. Used for stock price prediction. The Random Forest is set up as follows:
  * **Input:** Takes stock features (e.g., current price, market cap) as input.
  * **Ensemble of Trees:** Consists of multiple decision trees.
  * **Output:** Generates predictions by averaging the predictions of individual trees.
  * **Training:** The model uses the training data to learn patterns and relationships between features and stock prices.

* **Base Model (`base_model.py`):** An abstract base class that defines the common interface (`train`, `predict`, `evaluate`) for the machine learning models.

## Future Scope

The project has several potential areas for future development:

* **Integration of Real-Time Data:** Connect to a real-time stock data API to provide up-to-date predictions.
* **Advanced Technical Indicators:** Incorporate more advanced technical indicators, such as MACD, RSI, and Fibonacci retracements.
* **Sentiment Analysis:** Integrate sentiment analysis of news articles and social media to improve prediction accuracy.
* **Portfolio Management:** Add features for users to manage their virtual stock portfolios.
* **User Customization:** Allow users to customize the models and analysis based on their preferences.
* **Enhanced Visualization:** Improve the user interface and data visualization with more interactive charts and graphs.
* **Deployment:** Deploy the application to a production environment (e.g., Heroku, AWS, Google Cloud) to make it accessible to a wider audience.
* **Model Improvements:** Improve the model by adding more hyperparameters and changing the algorithm.


## Author
Abhay Sriwastav
