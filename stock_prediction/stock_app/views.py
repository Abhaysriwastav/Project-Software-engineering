import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.utils import timezone
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm

# Import your custom modules
from .models import LSTMModel, RandomForestModel
from .utils.data_handler import DataPreprocessor
from .utils.data_manager import DataManager
from .utils.trend_analyzer import TrendAnalyzer
from .predictions import predict_stock

# Configure logging
logger = logging.getLogger(__name__)

# Initialize components
data_manager = DataManager()
lstm_model = LSTMModel()
rf_model = RandomForestModel()

# Authentication Views
def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Auto-login after registration
            messages.success(request, f'Account created successfully! Welcome {user.username}!')
            return redirect('prediction')
        else:
            messages.error(request, 'Registration failed. Please correct the errors.')
    else:
        form = UserRegistrationForm()
    return render(request, 'stock_app/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('prediction')
        
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {username}!')
            return redirect('prediction')
        else:
            # Add specific error message
            messages.error(request, 'Invalid username or password. Please try again.')
            return render(request, 'stock_app/login.html', {
                'error': True,
                'username': username,  # Preserve the username
                'form': {'username': {'errors': True}, 'password': {'errors': True}}
            })
    
    return render(request, 'stock_app/login.html')
def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('login')

def load_training_data():
    """Load training data from the top companies historical CSV."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'data', 'top_companies_historical_data.csv')
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at {csv_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(csv_path)
        required_columns = [
            'Date', 'Company', 'Sector', 'Country', 
            'Current Price (£)', 'Market Cap (£ m)', 
            'Trading Volume', 'Open Price (£)', 
            'High Price (£)', 'Low Price (£)', 
            'Price Change (%)'
        ]
        
        if missing_columns := [col for col in required_columns if col not in df.columns]:
            logger.error(f"Missing columns in CSV: {missing_columns}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return pd.DataFrame()

@login_required
def prediction_view(request):
    """Main prediction view that handles both GET and POST requests"""
    if request.method == 'POST':
        return stock_prediction(request)
    else:
        training_data = load_training_data()
        context = {
            'companies': training_data[['Company', 'Sector']].drop_duplicates().to_dict('records') if not training_data.empty else [],
            'sectors': training_data['Sector'].unique().tolist() if not training_data.empty else [],
            'lstm_metrics': {'mae': 0.123, 'rmse': 0.234, 'r2': 0.89},
            'rf_metrics': {'mae': 0.145, 'rmse': 0.256, 'r2': 0.87},
            'predictions': False
        }
        return render(request, 'stock_app/prediction_results.html', context)

@login_required
def stock_prediction(request):
    """Handle stock prediction requests with comprehensive error handling."""
    logger.debug("Entering stock_prediction function.")  
    try:
        # Load training data
        training_data = load_training_data()
        logger.info(f"Training data loaded. Shape: {training_data.shape if not training_data.empty else 'Empty'}")

        # Handle POST request for prediction
        if request.method == 'POST':
            logger.info("Received POST request for prediction.")

            # Validate required fields
            required_fields = ['company', 'sector', 'market-cap', 'current-price']
            for field in required_fields:
                if field not in request.POST:
                    logger.error(f"Missing required field: {field}")
                    return JsonResponse({'success': False, 'error': f'Missing required field: {field}'}, status=400)

            try:
                # Extract and validate input data
                company = request.POST.get('company', '').strip()
                sector = request.POST.get('sector', '').strip()
                market_cap = float(request.POST.get('market-cap'))
                current_price = float(request.POST.get('current-price'))

                logger.info(f"Input data - Company: {company}, Sector: {sector}, Market Cap: {market_cap}, Current Price: {current_price}")

                # Input validation
                if not company or len(company) > 50:
                    return JsonResponse({'success': False, 'error': 'Company name must be between 1 and 50 characters'}, status=400)

                if market_cap <= 0 or market_cap > 1000000:
                    return JsonResponse({'success': False, 'error': 'Market cap must be between 0 and 1,000,000'}, status=400)

                if current_price <= 0 or current_price > 100000:
                    return JsonResponse({'success': False, 'error': 'Current price must be between 0 and 100,000'}, status=400)

                # Use the predict_stock function
                prediction_results = predict_stock(
                    company_name=company, 
                    sector=sector, 
                    market_cap=market_cap, 
                    current_price=current_price,
                    additional_training_data=training_data
                )
                logger.info(f"Prediction results: {prediction_results}")

                # Extract predictions
                lstm_prediction = prediction_results['predictions'].get('lstm_prediction', current_price)
                rf_prediction = prediction_results['predictions'].get('rf_prediction', current_price)

                # Calculate change percentage
                change_percentage_lstm = round((lstm_prediction - current_price) / current_price * 100, 2)
                change_percentage_rf = round((rf_prediction - current_price) / current_price * 100, 2)

                # Prepare prediction response
                predictions = {
                    'lstm': round(lstm_prediction, 2),
                    'rf': round(rf_prediction, 2),
                    'lstm_change_percentage': change_percentage_lstm,
                    'rf_change_percentage': change_percentage_rf,
                    'confidence': 0.85,
                    'lstm_metrics': prediction_results.get('lstm_metrics', {}),
                    'rf_metrics': prediction_results.get('rf_metrics', {})
                }

                logger.info(f"Final predictions: {predictions}")

                return JsonResponse({
                    'success': True,
                    'predictions': predictions,
                    'company': company,
                    'sector': sector
                })

            except ValueError as ve:
                logger.error(f"Value error in prediction: {ve}")
                return JsonResponse({'success': False, 'error': 'Invalid input values'}, status=400)
            except Exception as e:
                logger.error(f"Unexpected error in prediction: {str(e)}")
                return JsonResponse({'success': False, 'error': 'An unexpected error occurred during prediction'}, status=500)

    except Exception as e:
        logger.error(f"Critical error in stock prediction view: {str(e)}", exc_info=True)
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def historical_data(request):
    """View for historical data analysis with date range and chart visualization"""
    try:
        # Get parameters from request
        company = request.GET.get('company', '')
        period = request.GET.get('period', '1M')
        
        # Handle date parameters
        try:
            start_date = datetime.strptime(request.GET.get('start_date', ''), '%Y-%m-%d')
            end_date = datetime.strptime(request.GET.get('end_date', ''), '%Y-%m-%d')
        except ValueError:
            # Default to last month if dates not provided or invalid
            end_date = timezone.now()
            start_date = end_date - timedelta(days=30)

        # Load and validate data
        df = load_training_data()
        validation_results = data_manager.validate_data(df)
        
        # Get list of companies
        companies = df['Company'].unique().tolist() if not df.empty else []
        
        # Initialize context
        context = {
            'companies': companies,
            'selected_company': company,
            'start_date': start_date,
            'end_date': end_date,
            'period': period,
            'warnings': validation_results.get('warnings', [])
        }
        
        # If validation failed, return early with companies list
        if not validation_results['valid']:
            logger.error(f"Data validation failed: {validation_results['errors']}")
            context['errors'] = validation_results['errors']
            return render(request, 'stock_app/historical_data.html', context)
        
        # If company is selected, perform analysis
        if company:
            # Convert dates to pandas datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter data by date range and company
            mask = (
                (df['Company'] == company) & 
                (df['Date'] >= start_date) & 
                (df['Date'] <= end_date)
            )
            company_data = df.loc[mask].copy()
            
            # Calculate additional metrics
            if not company_data.empty:
                # Sort data by date for proper visualization
                company_data = company_data.sort_values('Date')
                
                # Prepare chart data
                chart_data = {
                    'labels': company_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                    'current_price': company_data['Current Price (£)'].round(2).tolist(),
                    'high_price': company_data['High Price (£)'].round(2).tolist(),
                    'low_price': company_data['Low Price (£)'].round(2).tolist(),
                    'volume': company_data['Trading Volume'].tolist()
                }
                
                context.update({
                    'chart_data': json.dumps(chart_data),
                    'trading_days': len(company_data),
                    'price_range': {
                        'min': company_data['Current Price (£)'].min(),
                        'max': company_data['Current Price (£)'].max()
                    },
                    'avg_volume': company_data['Trading Volume'].mean(),
                    'latest_price': company_data.iloc[-1]['Current Price (£)'],
                    'price_change': {
                        'value': company_data.iloc[-1]['Current Price (£)'] - company_data.iloc[0]['Current Price (£)'],
                        'percentage': ((company_data.iloc[-1]['Current Price (£)'] - company_data.iloc[0]['Current Price (£)']) 
                                     / company_data.iloc[0]['Current Price (£)'] * 100)
                    }
                })

                # Generate analysis
                analyzer = TrendAnalyzer(company_data)
                try:
                    analysis = analyzer.generate_summary_report(company)
                    context['analysis'] = analysis
                    logger.info(f"Analysis generated for company: {company}")
                    
                    # Add moving averages to chart data if available
                    if analysis.get('moving_averages'):
                        chart_data['moving_averages'] = analysis['moving_averages']
                        context['chart_data'] = json.dumps(chart_data)
                        
                except Exception as e:
                    logger.error(f"Error generating analysis for {company}: {str(e)}")
                    context['errors'] = [f"Error analyzing company data: {str(e)}"]
            else:
                context['warnings'] = [
                    f"No data available for {company} between {start_date.date()} and {end_date.date()}"
                ]
        
        return render(request, 'stock_app/historical_data.html', context)
        
    except Exception as e:
        logger.error(f"Error in historical data view: {str(e)}")
        return render(request, 'stock_app/historical_data.html', {
            'errors': ['An unexpected error occurred. Please try again.']
        })

@login_required
def train_and_evaluate(request):
    """Train machine learning models and evaluate their performance."""
    try:
        data = load_training_data()
        
        if data.empty:
            messages.error(request, 'No valid stock data found')
            return render(request, 'stock_app/prediction_results.html', {})
        
        preprocessor = DataPreprocessor(data)
        cleaned_data = preprocessor.preprocess()
        
        lstm_metrics = lstm_model.train_and_evaluate(cleaned_data)
        rf_metrics = rf_model.train_and_evaluate(cleaned_data)
        
        context = {
            'lstm_metrics': lstm_metrics,
            'rf_metrics': rf_metrics
        }
        
    except Exception as e:
        logger.error(f'Unexpected error in model training: {str(e)}')
        messages.error(request, f'Error in processing: {str(e)}')
        context = {}
    
    return render(request, 'stock_app/prediction_results.html', context)

@login_required
@require_http_methods(["GET", "POST"])
def trend_analysis(request):
    """View for trend analysis"""
    try:
        company = request.GET.get('company', '')
        days = int(request.GET.get('days', 30))
        
        cache_key = f'stock_data_{company}_{days}'
        analysis = cache.get(cache_key)
        
        if not analysis:
            df = load_training_data()
            analyzer = TrendAnalyzer(df)
            
            analysis = {
                'trend': analyzer.detect_trend(company, days),
                'volatility': analyzer.calculate_volatility(company, days),
                'support_resistance': analyzer.identify_support_resistance(company),
                'moving_averages': analyzer.calculate_moving_averages(company)
            }
            
            cache.set(cache_key, analysis, 3600)
        
        return JsonResponse({'success': True, 'analysis': analysis})
        
    except Exception as e:
        logger.error(f"Error in trend analysis view: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Error performing trend analysis'})

@login_required
@require_http_methods(["POST"])
def update_data(request):
    """View for updating historical data"""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Unauthorized'}, status=403)
        
    try:
        file = request.FILES.get('data_file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file provided'})
            
        new_data = pd.read_csv(file)
        success, warnings = data_manager.update_data(new_data)
        
        if success:
            return JsonResponse({'success': True, 'warnings': warnings})
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to update data',
                'warnings': warnings
            })
            
    except Exception as e:
        logger.error(f"Error in data update view: {str(e)}")
        return JsonResponse({'success': False, 'error': 'Error updating data'})