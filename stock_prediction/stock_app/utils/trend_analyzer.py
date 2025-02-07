import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

class TrendAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.data['Date'] = pd.to_datetime(self.data['Date'])

    def calculate_moving_averages(self, company, windows=[5, 20, 50]):
        """Calculate moving averages for different time windows"""
        company_data = self.data[self.data['Company'] == company].sort_values('Date')
        
        results = {'company': company, 'moving_averages': {}}
        for window in windows:
            ma = company_data['Current Price (£)'].rolling(window=window).mean()
            results['moving_averages'][f'MA{window}'] = ma.iloc[-1]
            
        return results

    def detect_trend(self, company, days=30):
        """Detect price trend for the specified number of days"""
        company_data = self.data[self.data['Company'] == company].sort_values('Date')
        recent_data = company_data.tail(days)
        
        # Calculate linear regression
        x = np.arange(len(recent_data))
        y = recent_data['Current Price (£)'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend strength and direction
        trend_strength = abs(r_value)
        trend_direction = 'up' if slope > 0 else 'down'
        
        return {
            'company': company,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }

    def calculate_volatility(self, company, days=30):
        """Calculate price volatility"""
        company_data = self.data[self.data['Company'] == company].sort_values('Date')
        recent_data = company_data.tail(days)
        
        daily_returns = recent_data['Current Price (£)'].pct_change()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        
        return {
            'company': company,
            'volatility': volatility,
            'daily_returns_std': daily_returns.std()
        }

    def identify_support_resistance(self, company, window=90):
        """Identify support and resistance levels"""
        company_data = self.data[self.data['Company'] == company].sort_values('Date')
        recent_data = company_data.tail(window)
        
        prices = recent_data['Current Price (£)']
        support = prices.min()
        resistance = prices.max()
        
        # Find price clusters using kernel density estimation
        kde = stats.gaussian_kde(prices)
        x_range = np.linspace(prices.min(), prices.max(), 100)
        density = kde(x_range)
        
        # Find local maxima in density
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i-1] and density[i] > density[i+1]:
                peaks.append(x_range[i])
        
        return {
            'company': company,
            'support': support,
            'resistance': resistance,
            'price_clusters': peaks
        }

    def generate_summary_report(self, company):
        """Generate comprehensive trend analysis report"""
        moving_averages = self.calculate_moving_averages(company)
        trend = self.detect_trend(company)
        volatility = self.calculate_volatility(company)
        support_resistance = self.identify_support_resistance(company)
        
        return {
            'company': company,
            'moving_averages': moving_averages['moving_averages'],
            'trend': {
                'direction': trend['trend_direction'],
                'strength': trend['trend_strength'],
                'confidence': 1 - trend['p_value']
            },
            'volatility': volatility['volatility'],
            'support_resistance': {
                'support': support_resistance['support'],
                'resistance': support_resistance['resistance'],
                'price_clusters': support_resistance['price_clusters']
            }
        }