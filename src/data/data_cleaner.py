"""
Data Cleaner Module for Carbon Emissions Forecasting System v3.0.

This module provides functionality to clean and preprocess emissions data
following IEEE standards for data quality assurance.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Data cleaner class for emissions data following IEEE standards.
    
    This class provides methods to clean, standardize, and prepare emissions data
    for machine learning models with comprehensive error handling and logging.
    """
    
    def __init__(self, config_path: Union[str, Path] = "config/config.json"):
        """
        Initialize the DataCleaner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.processed_data_dir = Path(self.config['data']['processed_dir'])
        self.entities = self.config['data']['entities']
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.cleaned_data = {}
        self.cleaning_log = []
        self.label_encoders = {}
        
        logger.info("DataCleaner initialized")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def clean_entity_data(self, df: pd.DataFrame, entity_name: str) -> pd.DataFrame:
        """
        Clean data for a specific entity.
        
        Args:
            df: Raw dataframe to clean
            entity_name: Name of the entity for logging purposes
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting data cleaning for {entity_name}")
        
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Track cleaning operations
        cleaning_operations = []
        
        try:
            # 1. Standardize column names
            df_clean, ops = self._standardize_column_names(df_clean)
            cleaning_operations.extend(ops)
            
            # 2. Handle missing values
            df_clean, ops = self._handle_missing_values(df_clean, entity_name)
            cleaning_operations.extend(ops)
            
            # 3. Clean and standardize date columns
            df_clean, ops = self._clean_date_columns(df_clean, entity_name)
            cleaning_operations.extend(ops)
            
            # 4. Clean and standardize emissions columns
            df_clean, ops = self._clean_emissions_columns(df_clean, entity_name)
            cleaning_operations.extend(ops)
            
            # 5. Clean and standardize sector columns
            df_clean, ops = self._clean_sector_columns(df_clean, entity_name)
            cleaning_operations.extend(ops)
            
            # 6. Remove duplicates
            df_clean, ops = self._remove_duplicates(df_clean)
            cleaning_operations.extend(ops)
            
            # 7. Sort by date if available
            df_clean, ops = self._sort_by_date(df_clean)
            cleaning_operations.extend(ops)
            
            # 8. Final validation
            df_clean, ops = self._final_validation(df_clean, entity_name)
            cleaning_operations.extend(ops)
            
            # Store cleaned data
            self.cleaned_data[entity_name] = df_clean
            
            # Log cleaning operations
            self.cleaning_log.append({
                'entity': entity_name,
                'timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'cleaned_shape': df_clean.shape,
                'operations': cleaning_operations
            })
            
            logger.info(f"Successfully cleaned {entity_name} data: {df.shape} -> {df_clean.shape}")
            return df_clean
            
        except Exception as e:
            error_msg = f"Error cleaning {entity_name} data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.cleaning_log.append({
                'entity': entity_name,
                'timestamp': datetime.now().isoformat(),
                'error': error_msg,
                'original_shape': df.shape
            })
            raise
    
    def _standardize_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Standardize column names."""
        operations = []
        original_columns = df.columns.tolist()
        
        # Convert to lowercase and replace spaces with underscores
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') 
                     for col in df.columns]
        
        # Remove special characters
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in df.columns]
        
        # Ensure unique column names
        df.columns = pd.Index([f"{col}_{i}" if df.columns.tolist()[:i].count(col) > 0 
                              else col for i, col in enumerate(df.columns)])
        
        operations.append(f"Standardized {len(original_columns)} column names")
        return df, operations
    
    def _handle_missing_values(self, df: pd.DataFrame, entity_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values in the dataframe."""
        operations = []
        
        # Count missing values before cleaning
        missing_before = df.isnull().sum().sum()
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        operations.append(f"Removed {len(df) - len(df.dropna(how='all'))} completely empty rows")
        
        # Handle missing values in date columns
        date_cols = [col for col in df.columns 
                    if any(keyword in col for keyword in ['date', 'month', 'year', 'time'])]
        for col in date_cols:
            if df[col].isnull().any():
                # Remove rows with missing dates as they're critical for time series
                df = df.dropna(subset=[col])
                operations.append(f"Removed rows with missing values in {col}")
        
        # Handle missing values in emissions columns
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col for keyword in ['emission', 'carbon', 'co2'])]
        for col in emissions_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                # For emissions, we might want to interpolate or use forward fill
                df[col] = df[col].interpolate(method='linear')
                operations.append(f"Interpolated {missing_count} missing values in {col}")
        
        # Handle missing values in sector columns
        sector_cols = [col for col in df.columns 
                      if any(keyword in col for keyword in ['sector', 'category'])]
        for col in sector_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna('Unknown')
                operations.append(f"Filled {missing_count} missing values in {col} with 'Unknown'")
        
        missing_after = df.isnull().sum().sum()
        operations.append(f"Reduced missing values from {missing_before} to {missing_after}")
        
        return df, operations
    
    def _clean_date_columns(self, df: pd.DataFrame, entity_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and standardize date columns."""
        operations = []
        
        date_cols = [col for col in df.columns 
                    if any(keyword in col for keyword in ['date', 'month', 'year', 'time'])]
        
        for col in date_cols:
            try:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Remove rows with invalid dates
                invalid_dates = df[col].isnull().sum()
                if invalid_dates > 0:
                    df = df.dropna(subset=[col])
                    operations.append(f"Removed {invalid_dates} rows with invalid dates in {col}")
                
                # Ensure dates are in reasonable range (e.g., 2000-2030)
                min_date = pd.Timestamp('2000-01-01')
                max_date = pd.Timestamp('2030-12-31')
                
                invalid_range = ((df[col] < min_date) | (df[col] > max_date)).sum()
                if invalid_range > 0:
                    df = df[(df[col] >= min_date) & (df[col] <= max_date)]
                    operations.append(f"Removed {invalid_range} rows with dates outside valid range in {col}")
                
                operations.append(f"Standardized date format for {col}")
                
            except Exception as e:
                operations.append(f"Error processing {col}: {str(e)}")
                logger.warning(f"Could not process date column {col}: {e}")
        
        return df, operations
    
    def _clean_emissions_columns(self, df: pd.DataFrame, entity_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and standardize emissions columns."""
        operations = []
        
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col for keyword in ['emission', 'carbon', 'co2'])]
        
        for col in emissions_cols:
            try:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove negative values (emissions should be positive)
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    df = df[df[col] >= 0]
                    operations.append(f"Removed {negative_count} negative values in {col}")
                
                # Handle outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    operations.append(f"Capped {outliers} outliers in {col}")
                
                operations.append(f"Standardized emissions data for {col}")
                
            except Exception as e:
                operations.append(f"Error processing {col}: {str(e)}")
                logger.warning(f"Could not process emissions column {col}: {e}")
        
        return df, operations
    
    def _clean_sector_columns(self, df: pd.DataFrame, entity_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """Clean and standardize sector columns."""
        operations = []
        
        sector_cols = [col for col in df.columns 
                      if any(keyword in col for keyword in ['sector', 'category'])]
        
        for col in sector_cols:
            try:
                # Convert to string and standardize
                df[col] = df[col].astype(str).str.strip().str.lower()
                
                # Remove empty strings
                empty_count = (df[col] == '').sum()
                if empty_count > 0:
                    df[col] = df[col].replace('', 'unknown')
                    operations.append(f"Replaced {empty_count} empty sector values with 'unknown'")
                
                # Standardize common sector names
                sector_mapping = {
                    'office': 'office',
                    'retail': 'retail',
                    'industrial': 'industrial',
                    'residential': 'residential',
                    'warehouse': 'warehouse',
                    'data_center': 'data_center',
                    'hospitality': 'hospitality',
                    'healthcare': 'healthcare',
                    'education': 'education'
                }
                
                df[col] = df[col].map(lambda x: sector_mapping.get(x, x))
                
                # Encode categorical values
                if entity_name not in self.label_encoders:
                    self.label_encoders[entity_name] = {}
                
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col])
                self.label_encoders[entity_name][col] = le
                
                operations.append(f"Standardized and encoded sector data for {col}")
                
            except Exception as e:
                operations.append(f"Error processing {col}: {str(e)}")
                logger.warning(f"Could not process sector column {col}: {e}")
        
        return df, operations
    
    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove duplicate rows."""
        operations = []
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            operations.append(f"Removed {duplicates} duplicate rows")
        else:
            operations.append("No duplicate rows found")
        
        return df, operations
    
    def _sort_by_date(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Sort dataframe by date if date column exists."""
        operations = []
        
        date_cols = [col for col in df.columns 
                    if any(keyword in col for keyword in ['date', 'month', 'year', 'time'])]
        
        if date_cols:
            df = df.sort_values(date_cols[0])
            operations.append(f"Sorted by {date_cols[0]}")
        else:
            operations.append("No date column found for sorting")
        
        return df, operations
    
    def _final_validation(self, df: pd.DataFrame, entity_name: str) -> Tuple[pd.DataFrame, List[str]]:
        """Perform final validation checks."""
        operations = []
        
        # Check for remaining missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            operations.append(f"Warning: {missing} missing values remain")
        else:
            operations.append("No missing values remain")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        operations.append(f"Found {len(numeric_cols)} numeric columns")
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            operations.append(f"Warning: {inf_count} infinite values found")
        
        return df, operations
    
    def save_cleaned_data(self, entity_name: str = None):
        """Save cleaned data to CSV files."""
        if entity_name:
            entities_to_save = [entity_name]
        else:
            entities_to_save = self.cleaned_data.keys()
        
        for entity in entities_to_save:
            if entity in self.cleaned_data:
                output_file = self.processed_data_dir / f"{entity}_cleaned.csv"
                self.cleaned_data[entity].to_csv(output_file, index=False)
                logger.info(f"Saved cleaned data for {entity} to {output_file}")
    
    def save_cleaning_log(self, output_file: str = "cleaning_log.json"):
        """Save cleaning log to file."""
        output_path = self.processed_data_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.cleaning_log, f, indent=2, default=str)
        
        logger.info(f"Cleaning log saved to {output_path}")
        return output_path
    
    def get_cleaning_summary(self) -> Dict:
        """Generate summary of cleaning operations."""
        summary = {
            'total_entities': len(self.entities),
            'cleaned_entities': len(self.cleaned_data),
            'total_operations': len(self.cleaning_log),
            'entities_summary': {}
        }
        
        for entity, df in self.cleaned_data.items():
            summary['entities_summary'][entity] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
        
        return summary


def main():
    """Main function for testing the DataCleaner."""
    from data_loader import DataLoader
    
    # Load raw data
    loader = DataLoader()
    raw_data = loader.load_all_entities()
    
    # Clean data
    cleaner = DataCleaner()
    
    for entity, df in raw_data.items():
        if df is not None:
            cleaned_df = cleaner.clean_entity_data(df, entity)
            print(f"\nCleaned {entity}: {df.shape} -> {cleaned_df.shape}")
    
    # Save results
    cleaner.save_cleaned_data()
    cleaner.save_cleaning_log()
    
    # Print summary
    summary = cleaner.get_cleaning_summary()
    print(f"\nCleaning Summary:")
    print(f"Entities processed: {summary['cleaned_entities']}")
    print(f"Total operations: {summary['total_operations']}")


if __name__ == "__main__":
    main() 