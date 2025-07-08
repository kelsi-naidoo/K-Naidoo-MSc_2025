"""
Data Loader Module for Carbon Emissions Forecasting System v3.0.

This module provides functionality to load and validate emissions data
from various sources following IEEE standards for data validation.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader class for emissions data following IEEE standards.
    
    This class provides methods to load, validate, and prepare emissions data
    from Excel files with proper error handling and logging.
    """
    
    def __init__(self, config_path: Union[str, Path] = "config/config.json"):
        """
        Initialize the DataLoader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.raw_data_dir = Path(self.config['data']['raw_dir'])
        self.processed_data_dir = Path(self.config['data']['processed_dir'])
        self.entities = self.config['data']['entities']
        
        # Create directories if they don't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.raw_data = {}
        self.loading_errors = []
        
        logger.info(f"DataLoader initialized with {len(self.entities)} entities")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def load_entity_data(self, entity_name: str) -> Optional[pd.DataFrame]:
        """
        Load data for a specific entity.
        
        Args:
            entity_name: Name of the entity to load data for
            
        Returns:
            DataFrame containing the entity's data or None if loading fails
        """
        try:
            # Find the emissions file for the entity
            file_pattern = self.config['data']['file_patterns']['emissions']
            file_path = self.raw_data_dir / file_pattern.replace('*', entity_name)
            
            if not file_path.exists():
                error_msg = f"File not found for {entity_name}: {file_path}"
                logger.error(error_msg)
                self.loading_errors.append({
                    'entity': entity_name,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                return None
            
            logger.info(f"Loading data from {file_path}")
            
            # Read the Excel file
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Basic validation
            if df.empty:
                error_msg = f"Empty dataframe loaded for {entity_name}"
                logger.warning(error_msg)
                self.loading_errors.append({
                    'entity': entity_name,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                })
                return None
            
            # Store raw data
            self.raw_data[entity_name] = df.copy()
            
            logger.info(f"Successfully loaded {entity_name} data: {df.shape}")
            return df
            
        except Exception as e:
            error_msg = f"Error loading {entity_name} data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.loading_errors.append({
                'entity': entity_name,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def load_all_entities(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all entities.
        
        Returns:
            Dictionary mapping entity names to their dataframes
        """
        logger.info("Loading data for all entities...")
        
        loaded_data = {}
        
        for entity in self.entities:
            df = self.load_entity_data(entity)
            if df is not None:
                loaded_data[entity] = df
        
        logger.info(f"Loaded data for {len(loaded_data)} entities")
        
        if self.loading_errors:
            logger.warning(f"Encountered {len(self.loading_errors)} loading errors")
            self._save_loading_errors()
        
        return loaded_data
    
    def get_data_summary(self) -> Dict:
        """
        Generate summary statistics for loaded data.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_entities': len(self.entities),
            'loaded_entities': len(self.raw_data),
            'loading_errors': len(self.loading_errors),
            'entities_data': {}
        }
        
        for entity, df in self.raw_data.items():
            entity_summary = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Add date range if date column exists
            date_cols = [col for col in df.columns 
                        if any(keyword in col.lower() 
                              for keyword in ['date', 'month', 'year', 'time'])]
            if date_cols:
                try:
                    date_col = date_cols[0]
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    valid_dates = df[date_col].dropna()
                    if not valid_dates.empty:
                        entity_summary['date_range'] = {
                            'start': valid_dates.min().isoformat(),
                            'end': valid_dates.max().isoformat(),
                            'total_days': (valid_dates.max() - valid_dates.min()).days
                        }
                except Exception as e:
                    logger.warning(f"Could not process dates for {entity}: {e}")
            
            summary['entities_data'][entity] = entity_summary
        
        return summary
    
    def _save_loading_errors(self):
        """Save loading errors to file for debugging."""
        if self.loading_errors:
            error_file = self.processed_data_dir / 'loading_errors.json'
            with open(error_file, 'w') as f:
                json.dump(self.loading_errors, f, indent=2)
            logger.info(f"Loading errors saved to {error_file}")
    
    def save_raw_data_summary(self, output_file: str = "raw_data_summary.json"):
        """Save data summary to file."""
        summary = self.get_data_summary()
        output_path = self.processed_data_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Data summary saved to {output_path}")
        return output_path
    
    def validate_data_structure(self) -> Dict[str, List[str]]:
        """
        Validate data structure against expected schema.
        
        Returns:
            Dictionary containing validation results for each entity
        """
        validation_results = {}
        
        expected_columns = {
            'temporal': ['date', 'month', 'year', 'time'],
            'emissions': ['emission', 'carbon', 'co2'],
            'sector': ['sector', 'category', 'division']
        }
        
        for entity, df in self.raw_data.items():
            entity_validation = {
                'temporal_columns': [],
                'emissions_columns': [],
                'sector_columns': [],
                'missing_required': []
            }
            
            # Check for temporal columns
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in expected_columns['temporal']):
                    entity_validation['temporal_columns'].append(col)
                elif any(keyword in col_lower for keyword in expected_columns['emissions']):
                    entity_validation['emissions_columns'].append(col)
                elif any(keyword in col_lower for keyword in expected_columns['sector']):
                    entity_validation['sector_columns'].append(col)
            
            # Check for missing required columns
            if not entity_validation['temporal_columns']:
                entity_validation['missing_required'].append('temporal_column')
            if not entity_validation['emissions_columns']:
                entity_validation['missing_required'].append('emissions_column')
            
            validation_results[entity] = entity_validation
        
        return validation_results


def main():
    """Main function for testing the DataLoader."""
    # Initialize loader
    loader = DataLoader()
    
    # Load all entities
    data = loader.load_all_entities()
    
    # Print summary
    summary = loader.get_data_summary()
    print(f"\nData Loading Summary:")
    print(f"Total entities: {summary['total_entities']}")
    print(f"Loaded entities: {summary['loaded_entities']}")
    print(f"Loading errors: {summary['loading_errors']}")
    
    # Validate structure
    validation = loader.validate_data_structure()
    print(f"\nValidation Results:")
    for entity, results in validation.items():
        print(f"\n{entity}:")
        print(f"  Temporal columns: {results['temporal_columns']}")
        print(f"  Emissions columns: {results['emissions_columns']}")
        print(f"  Sector columns: {results['sector_columns']}")
        if results['missing_required']:
            print(f"  Missing required: {results['missing_required']}")
    
    # Save summary
    loader.save_raw_data_summary()


if __name__ == "__main__":
    main() 