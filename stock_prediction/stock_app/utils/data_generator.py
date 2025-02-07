import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class StockDataGenerator:
    def __init__(self):
        self.sectors = {
            'Technology': [
                'Software', 'Hardware', 'IT Services', 'Semiconductors', 'Cloud Computing'
            ],
            'Finance': [
                'Banking', 'Insurance', 'Investment Services', 'FinTech', 'Asset Management'
            ],
            'Healthcare': [
                'Pharmaceuticals', 'Medical Devices', 'Healthcare Services', 'Biotechnology'
            ],
            'Energy': [
                'Oil & Gas', 'Renewable Energy', 'Utilities', 'Energy Equipment'
            ]
        }
        
        self.sector_metrics = {
            'Technology': {'base_price': 500, 'volatility': 0.3},
            'Finance': {'base_price': 300, 'volatility': 0.2},
            'Healthcare': {'base_price': 400, 'volatility': 0.25},
            'Energy': {'base_price': 200, 'volatility': 0.35}
        }

    def generate_company_data(self, num_companies=100):
        data = []
        for _ in range(num_companies):
            sector = np.random.choice(list(self.sectors.keys()))
            subsector = np.random.choice(self.sectors[sector])
            metrics = self.sector_metrics[sector]
            
            company_data = {
                'Company': f"{sector[:3]}{subsector.split()[0]}Corp{_}",
                'Sector': sector,
                'Subsector': subsector,
                'Market Cap (£ m)': round(np.random.uniform(100, 10000), 2),
                'Current Price per Share (pence)': round(np.random.normal(
                    metrics['base_price'], 
                    metrics['base_price'] * metrics['volatility']
                ), 2),
                'Last Statement Year': 2023
            }
            data.append(company_data)
            
        return pd.DataFrame(data)