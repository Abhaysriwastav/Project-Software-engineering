import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        """
        Initialize DataPreprocessor with input data
        
        Args:
            data (pandas.DataFrame): Input data for preprocessing
        """
        # Create a deep copy to avoid SettingWithCopyWarning
        self.data = data.copy()
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        
        Returns:
            pandas.DataFrame: Cleaned dataset
        """
        # Handle string columns
        string_columns = ['Company', 'Sector']
        for col in string_columns:
            if col in self.data.columns:
                # Use .loc to avoid SettingWithCopyWarning
                self.data.loc[:, col] = self.data[col].fillna('Unknown')
        
        # Handle Subsector if it exists
        if 'Subsector' in self.data.columns:
            self.data.loc[:, 'Subsector'] = self.data['Subsector'].fillna('Unknown')
        
        # Handle numeric columns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Use median for numeric columns
            self.data.loc[:, col] = self.data[col].fillna(self.data[col].median())
        
        return self.data
    
    def remove_outliers(self):
        """
        Remove outliers using IQR method
        
        Returns:
            pandas.DataFrame: Dataset with outliers removed
        """
        # Create a copy to avoid modifying the original dataframe
        cleaned_data = self.data.copy()
        
        # Identify numeric columns
        numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Calculate IQR
            Q1 = cleaned_data[col].quantile(0.25)
            Q3 = cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers
            cleaned_data = cleaned_data[
                (cleaned_data[col] >= lower_bound) & 
                (cleaned_data[col] <= upper_bound)
            ]
        
        return cleaned_data
    
    def preprocess(self):
        """
        Complete preprocessing pipeline
        
        Returns:
            pandas.DataFrame: Fully preprocessed dataset
        """
        # Handle missing values
        self.data = self.handle_missing_values()
        
        # Remove outliers
        self.data = self.remove_outliers()
        
        return self.data