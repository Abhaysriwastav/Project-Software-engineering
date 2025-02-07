import os
import shutil
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

class DataManager:
    def __init__(self, data_dir='data', backup_dir='data/backups'):
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True, parents=True)

    def backup_data(self, filename='top_companies_historical_data.csv'):
        """Create a timestamped backup of the data file"""
        try:
            source = self.data_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Source file {filename} not found")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filename.split('.')[0]}_{timestamp}.csv"
            destination = self.backup_dir / backup_name

            shutil.copy2(source, destination)
            self.logger.info(f"Backup created: {backup_name}")
            return True
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            return False

    def validate_data(self, df):
        """Validate data structure and content"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Required columns
        required_columns = [
            'Date', 'Company', 'Sector', 'Country', 'Current Price (£)',
            'Market Cap (£ m)', 'Trading Volume', 'Open Price (£)',
            'High Price (£)', 'Low Price (£)', 'Price Change (%)'
        ]

        # Check columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_columns}")

        # Data type validation
        if 'valid' in validation_results and validation_results['valid']:
            try:
                # Numeric columns validation
                numeric_columns = [
                    'Current Price (£)', 'Market Cap (£ m)', 'Trading Volume',
                    'Open Price (£)', 'High Price (£)', 'Low Price (£)', 'Price Change (%)'
                ]
                for col in numeric_columns:
                    non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
                    if non_numeric > 0:
                        validation_results['warnings'].append(
                            f"{non_numeric} non-numeric values found in {col}"
                        )

                # Date validation
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                invalid_dates = df['Date'].isna().sum()
                if invalid_dates > 0:
                    validation_results['warnings'].append(
                        f"{invalid_dates} invalid dates found"
                    )

                # Logical validations
                price_errors = df[df['Low Price (£)'] > df['High Price (£)']].shape[0]
                if price_errors > 0:
                    validation_results['warnings'].append(
                        f"{price_errors} records where Low Price exceeds High Price"
                    )

            except Exception as e:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Validation error: {str(e)}")

        return validation_results

    def update_data(self, new_data_df, filename='top_companies_historical_data.csv'):
        """Update the dataset with new data"""
        try:
            # Create backup before update
            self.backup_data(filename)

            # Validate new data
            validation_results = self.validate_data(new_data_df)
            if not validation_results['valid']:
                raise ValueError(f"Invalid data: {validation_results['errors']}")

            # Save new data
            filepath = self.data_dir / filename
            new_data_df.to_csv(filepath, index=False)
            self.logger.info(f"Data updated successfully: {filename}")

            # Log any warnings
            for warning in validation_results['warnings']:
                self.logger.warning(warning)

            return True, validation_results['warnings']
        except Exception as e:
            self.logger.error(f"Update failed: {str(e)}")
            return False, [str(e)]