"""
Test module for data processing functionality.

This module contains unit tests for the data processing components
following IEEE standards for software testing.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import json

# Add src to path
sys.path.append('../src')

from data import DataLoader, DataCleaner, DataValidator


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.raw_dir.mkdir()
        
        # Create test configuration
        self.config = {
            "data": {
                "raw_dir": str(self.raw_dir),
                "processed_dir": str(Path(self.temp_dir) / "processed"),
                "entities": ["TestEntity"],
                "file_patterns": {
                    "emissions": "Emissions_*.xlsx"
                }
            }
        }
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create test emissions data."""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        data = {
            'Month': dates,
            'Emissions': np.random.normal(1000, 200, 24),
            'Sector': np.random.choice(['Office', 'Retail', 'Industrial'], 24)
        }
        
        df = pd.DataFrame(data)
        
        # Save to Excel file
        test_file = self.raw_dir / "Emissions_TestEntity.xlsx"
        df.to_excel(test_file, index=False)
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        # Save config to temporary file
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        loader = DataLoader(str(config_file))
        self.assertIsNotNone(loader)
        self.assertEqual(len(loader.entities), 1)
        self.assertEqual(loader.entities[0], "TestEntity")
    
    def test_load_entity_data(self):
        """Test loading entity data."""
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        loader = DataLoader(str(config_file))
        df = loader.load_entity_data("TestEntity")
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 24)
        self.assertIn('Month', df.columns)
        self.assertIn('Emissions', df.columns)
        self.assertIn('Sector', df.columns)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "data": {
                "processed_dir": str(Path(self.temp_dir) / "processed")
            }
        }
        
        # Create test data with issues
        self.test_data = pd.DataFrame({
            'Month': pd.date_range('2020-01-01', periods=10, freq='M'),
            'Emissions': [1000, 1100, np.nan, 1200, 1300, 1400, 1500, 1600, 1700, 1800],
            'Sector': ['Office', 'Retail', 'Industrial', 'Office', 'Retail', 
                      'Industrial', 'Office', 'Retail', 'Industrial', 'Office']
        })
    
    def test_data_cleaner_initialization(self):
        """Test DataCleaner initialization."""
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        cleaner = DataCleaner(str(config_file))
        self.assertIsNotNone(cleaner)
    
    def test_clean_entity_data(self):
        """Test data cleaning functionality."""
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        cleaner = DataCleaner(str(config_file))
        cleaned_df = cleaner.clean_entity_data(self.test_data, "TestEntity")
        
        self.assertIsNotNone(cleaned_df)
        self.assertLess(len(cleaned_df), len(self.test_data))  # Should remove NaN row
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0)  # No missing values
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "data": {
                "processed_dir": str(Path(self.temp_dir) / "processed")
            }
        }
        
        # Create clean test data
        self.clean_data = pd.DataFrame({
            'Month': pd.date_range('2020-01-01', periods=12, freq='M'),
            'Emissions': np.random.normal(1000, 200, 12),
            'Sector': np.random.choice(['Office', 'Retail', 'Industrial'], 12)
        })
    
    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        validator = DataValidator(str(config_file))
        self.assertIsNotNone(validator)
    
    def test_validate_entity_data(self):
        """Test data validation functionality."""
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        validator = DataValidator(str(config_file))
        validation_result = validator.validate_entity_data(self.clean_data, "TestEntity")
        
        self.assertIsNotNone(validation_result)
        self.assertIn('overall_status', validation_result)
        self.assertIn('checks', validation_result)
        self.assertIn('errors', validation_result)
        self.assertIn('warnings', validation_result)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 