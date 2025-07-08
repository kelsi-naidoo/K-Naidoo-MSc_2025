"""
Data Validator Module for Carbon Emissions Forecasting System v3.0.

This module provides functionality to validate data quality and integrity
following IEEE standards for data validation and verification.

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
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data validator class for emissions data following IEEE standards.
    
    This class provides comprehensive validation methods to ensure data quality,
    integrity, and suitability for machine learning models.
    """
    
    def __init__(self, config_path: Union[str, Path] = "config/config.json"):
        """
        Initialize the DataValidator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.processed_data_dir = Path(self.config['data']['processed_dir'])
        self.entities = self.config['data']['entities']
        
        # Validation results storage
        self.validation_results = {}
        self.validation_errors = []
        
        logger.info("DataValidator initialized")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def validate_entity_data(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """
        Perform comprehensive validation on entity data.
        
        Args:
            df: Dataframe to validate
            entity_name: Name of the entity for logging
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Starting validation for {entity_name}")
        
        validation_result = {
            'entity': entity_name,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'checks': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # 1. Basic structure validation
            structure_result = self._validate_structure(df, entity_name)
            validation_result['checks']['structure'] = structure_result
            
            # 2. Data type validation
            dtype_result = self._validate_data_types(df, entity_name)
            validation_result['checks']['data_types'] = dtype_result
            
            # 3. Missing values validation
            missing_result = self._validate_missing_values(df, entity_name)
            validation_result['checks']['missing_values'] = missing_result
            
            # 4. Temporal validation
            temporal_result = self._validate_temporal_data(df, entity_name)
            validation_result['checks']['temporal'] = temporal_result
            
            # 5. Emissions data validation
            emissions_result = self._validate_emissions_data(df, entity_name)
            validation_result['checks']['emissions'] = emissions_result
            
            # 6. Sector data validation
            sector_result = self._validate_sector_data(df, entity_name)
            validation_result['checks']['sectors'] = sector_result
            
            # 7. Statistical validation
            stats_result = self._validate_statistical_properties(df, entity_name)
            validation_result['checks']['statistics'] = stats_result
            
            # 8. Business logic validation
            business_result = self._validate_business_logic(df, entity_name)
            validation_result['checks']['business_logic'] = business_result
            
            # Determine overall status
            validation_result = self._determine_overall_status(validation_result)
            
            # Store results
            self.validation_results[entity_name] = validation_result
            
            logger.info(f"Validation completed for {entity_name}: {validation_result['overall_status']}")
            return validation_result
            
        except Exception as e:
            error_msg = f"Error during validation of {entity_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            validation_result['overall_status'] = 'ERROR'
            validation_result['errors'].append(error_msg)
            self.validation_errors.append(validation_result)
            return validation_result
    
    def _validate_structure(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate basic data structure."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        # Check if dataframe is empty
        if df.empty:
            result['status'] = 'FAIL'
            result['errors'].append("Dataframe is empty")
            return result
        
        # Check minimum required columns - updated for standardized names
        required_columns = {
            'temporal': ['fiscalyear', 'date', 'month', 'year', 'time'],
            'emissions': ['emissions', 'emission', 'carbon', 'co2'],
            'sector': ['sector', 'property_type', 'category']
        }
        
        found_columns = {
            'temporal': [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in required_columns['temporal'])],
            'emissions': [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in required_columns['emissions'])],
            'sector': [col for col in df.columns 
                      if any(keyword in col.lower() for keyword in required_columns['sector'])]
        }
        
        result['details']['found_columns'] = found_columns
        result['details']['total_columns'] = len(df.columns)
        result['details']['total_rows'] = len(df)
        
        # Check for required column types
        if not found_columns['temporal']:
            result['warnings'].append("No temporal column found")
        if not found_columns['emissions']:
            result['status'] = 'FAIL'
            result['errors'].append("No emissions column found")
        if not found_columns['sector']:
            result['warnings'].append("No sector column found")
        
        return result
    
    def _validate_data_types(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate data types of columns."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        result['details']['dtypes'] = df.dtypes.to_dict()
        
        # Check temporal columns - updated for standardized names
        temporal_cols = [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in ['fiscalyear', 'date', 'month', 'year', 'time'])]
        
        for col in temporal_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                result['warnings'].append(f"Temporal column {col} is not datetime type")
        
        # Check emissions columns - only numeric ones, not categorical like emissions_source
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emissions', 'emission', 'carbon', 'co2'])
                         and col != 'emissions_source'  # Exclude categorical emissions source
                         and pd.api.types.is_numeric_dtype(df[col])]  # Only numeric columns
        
        for col in emissions_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                result['status'] = 'FAIL'
                result['errors'].append(f"Emissions column {col} is not numeric type")
        
        return result
    
    def _validate_missing_values(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate missing values."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        result['details']['missing_counts'] = missing_counts.to_dict()
        result['details']['missing_percentages'] = missing_percentages.to_dict()
        
        # Check for critical missing values - updated for standardized names
        critical_columns = []
        temporal_cols = [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in ['fiscalyear', 'date', 'month', 'year', 'time'])]
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emissions', 'emission', 'carbon', 'co2'])]
        critical_columns.extend(temporal_cols)
        critical_columns.extend(emissions_cols)
        
        for col in critical_columns:
            if col in missing_counts and missing_counts[col] > 0:
                missing_pct = missing_percentages[col]
                if missing_pct > 50:
                    result['status'] = 'FAIL'
                    result['errors'].append(f"Critical column {col} has {missing_pct:.1f}% missing values")
                elif missing_pct > 10:
                    result['warnings'].append(f"Column {col} has {missing_pct:.1f}% missing values")
        
        return result
    
    def _validate_temporal_data(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate temporal data."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        # Find temporal columns - updated for standardized names
        temporal_cols = [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in ['fiscalyear', 'date', 'month', 'year', 'time'])]
        
        if not temporal_cols:
            result['warnings'].append("No temporal columns found")
            return result
        
        for col in temporal_cols:
            try:
                # Ensure datetime type
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Check for valid dates
                valid_dates = df[col].notna().sum()
                total_dates = len(df[col])
                
                if valid_dates == 0:
                    result['status'] = 'FAIL'
                    result['errors'].append(f"No valid dates found in {col}")
                elif valid_dates < total_dates * 0.9:  # Less than 90% valid
                    result['warnings'].append(f"Column {col} has {total_dates - valid_dates} invalid dates")
                
                # Check date range
                if valid_dates > 0:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    
                    # Check for reasonable date range (2000-2030)
                    if min_date < pd.Timestamp('2000-01-01'):
                        result['warnings'].append(f"Column {col} has dates before 2000: {min_date}")
                    if max_date > pd.Timestamp('2030-12-31'):
                        result['warnings'].append(f"Column {col} has dates after 2030: {max_date}")
                    
                    result['details'][f'{col}_range'] = {
                        'min': min_date.isoformat(),
                        'max': max_date.isoformat(),
                        'total_days': (max_date - min_date).days
                    }
                
            except Exception as e:
                result['errors'].append(f"Error processing temporal column {col}: {str(e)}")
        
        return result
    
    def _validate_emissions_data(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate emissions data."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        # Find emissions columns - only numeric ones, not categorical like emissions_source
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emissions', 'emission', 'carbon', 'co2'])
                         and col != 'emissions_source'  # Exclude categorical emissions source
                         and pd.api.types.is_numeric_dtype(df[col])]  # Only numeric columns
        
        if not emissions_cols:
            result['status'] = 'FAIL'
            result['errors'].append("No numeric emissions columns found")
            return result
        
        for col in emissions_cols:
            try:
                # Basic statistics
                stats = df[col].describe()
                result['details'][f'{col}_stats'] = {
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    '25%': stats['25%'],
                    '50%': stats['50%'],
                    '75%': stats['75%'],
                    'max': stats['max']
                }
                
                # Check for negative values (corrections are expected)
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    result['warnings'].append(f"Column {col} has {negative_count} negative values (corrections)")
                
                # Check for zero values
                zero_count = (df[col] == 0).sum()
                if zero_count > len(df) * 0.5:  # More than 50% zeros
                    result['warnings'].append(f"Column {col} has {zero_count} zero values ({zero_count/len(df)*100:.1f}%)")
                
                # Check for outliers using IQR method
                Q1 = stats['25%']
                Q3 = stats['75%']
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > len(df) * 0.1:  # More than 10% outliers
                    result['warnings'].append(f"Column {col} has {outliers} outliers ({outliers/len(df)*100:.1f}%)")
                
            except Exception as e:
                result['errors'].append(f"Error processing emissions column {col}: {str(e)}")
        
        return result
    
    def _validate_sector_data(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate sector data properties."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        sector_cols = [col for col in df.columns 
                      if any(keyword in col for keyword in ['sector', 'property_type', 'category'])]
        
        if not sector_cols:
            result['warnings'].append("No sector columns found")
            return result
        
        for col in sector_cols:
            try:
                # Count unique sectors
                unique_sectors = df[col].nunique()
                sector_counts = df[col].value_counts()
                
                result['details'][col] = {
                    'unique_count': unique_sectors,
                    'sector_distribution': sector_counts.to_dict()
                }
                
                # Check for too many unique sectors
                if unique_sectors > 20:
                    result['warnings'].append(f"Column {col} has many unique sectors: {unique_sectors}")
                
                # Check for imbalanced sectors
                max_sector_count = sector_counts.max()
                min_sector_count = sector_counts.min()
                if min_sector_count < max_sector_count * 0.01:  # Less than 1% of max
                    result['warnings'].append(f"Column {col} has imbalanced sector distribution")
                
                # Check for unknown/empty sectors
                unknown_count = (df[col].isin(['unknown', 'Unknown', '', 'nan'])).sum()
                if unknown_count > len(df) * 0.1:  # More than 10% unknown
                    result['warnings'].append(f"Column {col} has {unknown_count} unknown sector values")
                
            except Exception as e:
                result['errors'].append(f"Error analyzing {col}: {str(e)}")
        
        return result
    
    def _validate_statistical_properties(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate statistical properties of the data."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        # Check for normality in emissions data
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col for keyword in ['emission', 'carbon', 'co2'])]
        
        for col in emissions_cols:
            try:
                # Remove missing values for statistical tests
                clean_data = df[col].dropna()
                
                if len(clean_data) > 3:
                    # Shapiro-Wilk test for normality
                    shapiro_stat, shapiro_p = stats.shapiro(clean_data)
                    
                    result['details'][col] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                    
                    if shapiro_p <= 0.05:
                        result['warnings'].append(f"Column {col} may not be normally distributed (p={shapiro_p:.4f})")
                
            except Exception as e:
                result['warnings'].append(f"Could not perform statistical tests on {col}: {str(e)}")
        
        return result
    
    def _validate_business_logic(self, df: pd.DataFrame, entity_name: str) -> Dict:
        """Validate business logic and domain-specific rules."""
        result = {
            'status': 'PASS',
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        # Check for reasonable emissions values based on sector
        sector_cols = [col for col in df.columns 
                      if any(keyword in col for keyword in ['sector', 'property_type', 'category'])]
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col for keyword in ['emission', 'carbon', 'co2'])]
        
        if sector_cols and emissions_cols:
            sector_col = sector_cols[0]
            emissions_col = emissions_cols[0]
            
            try:
                # Check sector-specific emissions ranges
                sector_stats = df.groupby(sector_col)[emissions_col].agg(['mean', 'std', 'min', 'max'])
                
                result['details']['sector_emissions'] = sector_stats.to_dict()
                
                # Flag sectors with unusually high or low emissions
                for sector in sector_stats.index:
                    mean_emissions = sector_stats.loc[sector, 'mean']
                    if mean_emissions > 10000:  # Very high emissions
                        result['warnings'].append(f"Sector {sector} has high average emissions: {mean_emissions:.2f}")
                    elif mean_emissions < 1:  # Very low emissions
                        result['warnings'].append(f"Sector {sector} has low average emissions: {mean_emissions:.2f}")
                
            except Exception as e:
                result['warnings'].append(f"Could not perform business logic validation: {str(e)}")
        
        return result
    
    def _determine_overall_status(self, validation_result: Dict) -> Dict:
        """Determine overall validation status based on individual checks."""
        has_errors = False
        has_warnings = False
        
        for check_name, check_result in validation_result['checks'].items():
            if check_result['status'] == 'FAIL':
                has_errors = True
            elif check_result['status'] == 'WARNING':
                has_warnings = True
        
        if has_errors:
            validation_result['overall_status'] = 'FAIL'
        elif has_warnings:
            validation_result['overall_status'] = 'WARNING'
        else:
            validation_result['overall_status'] = 'PASS'
        
        return validation_result
    
    def save_validation_results(self, output_file: str = "validation_results.json"):
        """Save validation results to file."""
        output_path = self.processed_data_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved to {output_path}")
        return output_path
    
    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report_lines = [
            "=" * 80,
            "DATA VALIDATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for entity, result in self.validation_results.items():
            report_lines.extend([
                f"Entity: {entity}",
                f"Status: {result['overall_status']}",
                f"Timestamp: {result['timestamp']}",
                ""
            ])
            
            if result['errors']:
                report_lines.append("ERRORS:")
                for error in result['errors']:
                    report_lines.append(f"  • {error}")
                report_lines.append("")
            
            if result['warnings']:
                report_lines.append("WARNINGS:")
                for warning in result['warnings']:
                    report_lines.append(f"  • {warning}")
                report_lines.append("")
            
            report_lines.append("-" * 40)
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.processed_data_dir / "validation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Validation report saved to {report_file}")
        return report


def main():
    """Main function for testing the DataValidator."""
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    raw_data = loader.load_all_entities()
    
    cleaner = DataCleaner()
    cleaned_data = {}
    
    for entity, df in raw_data.items():
        if df is not None:
            cleaned_df = cleaner.clean_entity_data(df, entity)
            cleaned_data[entity] = cleaned_df
    
    # Validate cleaned data
    validator = DataValidator()
    
    for entity, df in cleaned_data.items():
        validation_result = validator.validate_entity_data(df, entity)
        print(f"\nValidation for {entity}: {validation_result['overall_status']}")
    
    # Save results and generate report
    validator.save_validation_results()
    report = validator.generate_validation_report()
    print("\n" + report)


if __name__ == "__main__":
    main() 