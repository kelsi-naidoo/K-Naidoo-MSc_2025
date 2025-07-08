"""
Main script for Carbon Emissions Forecasting System v3.0.

This script orchestrates the complete data processing pipeline
following IEEE standards for software engineering.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import argparse
import logging
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data import DataLoader, DataCleaner, DataValidator
from models import AnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reports/logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(description='Carbon Emissions Forecasting System v3.0')
    parser.add_argument('--config', type=str, default='../config/config.json',
                       help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['load', 'clean', 'validate', 'anomaly', 'all'],
                       default='all', help='Processing step to run')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting Carbon Emissions Forecasting System v3.0")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Processing step: {args.step}")
        
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")
        
        if args.step in ['load', 'all']:
            logger.info("Step 1: Loading data...")
            
            # Initialize data loader
            loader = DataLoader(args.config)
            
            # Load all entities
            raw_data = loader.load_all_entities()
            
            # Generate and save summary
            summary = loader.get_data_summary()
            loader.save_raw_data_summary()
            
            logger.info(f"Loaded data for {summary['loaded_entities']} entities")
            
            if summary['loading_errors'] > 0:
                logger.warning(f"Encountered {summary['loading_errors']} loading errors")
        
        if args.step in ['clean', 'all']:
            logger.info("Step 2: Cleaning data...")
            
            # Initialize data cleaner
            cleaner = DataCleaner(args.config)
            
            # Clean data for each entity
            for entity, df in raw_data.items():
                if df is not None:
                    logger.info(f"Cleaning data for {entity}...")
                    cleaned_df = cleaner.clean_entity_data(df, entity)
                    logger.info(f"Cleaned {entity}: {df.shape} -> {cleaned_df.shape}")
            
            # Save cleaned data
            cleaner.save_cleaned_data()
            cleaner.save_cleaning_log()
            
            # Generate cleaning summary
            cleaning_summary = cleaner.get_cleaning_summary()
            logger.info(f"Cleaned data for {cleaning_summary['cleaned_entities']} entities")
        
        if args.step in ['validate', 'all']:
            logger.info("Step 3: Validating data...")
            
            # Initialize data validator
            validator = DataValidator(args.config)
            
            # Validate cleaned data
            validation_results = {}
            for entity, df in cleaned_data.items():
                logger.info(f"Validating data for {entity}...")
                validation_result = validator.validate_entity_data(df, entity)
                validation_results[entity] = validation_result
                
                status = validation_result['overall_status']
                errors = len(validation_result['errors'])
                warnings = len(validation_result['warnings'])
                
                logger.info(f"Validation for {entity}: {status} (Errors: {errors}, Warnings: {warnings})")
            
            # Save validation results
            validator.save_validation_results()
            
            # Generate validation report
            report = validator.generate_validation_report()
            logger.info("Validation report generated")
        
        if args.step in ['anomaly', 'all']:
            logger.info("Step 4: Anomaly Detection...")
            
            # Initialize anomaly detector
            detector = AnomalyDetector(args.config)
            
            # Load cleaned data for anomaly detection
            cleaned_data = detector.load_cleaned_data()
            
            # Detect anomalies for each entity
            for entity, df in cleaned_data.items():
                if df is not None:
                    logger.info(f"Detecting anomalies for {entity}...")
                    anomalies = detector.detect_anomalies(df, entity)
                    logger.info(f"Found {len(anomalies)} anomalies in {entity}")
                    
                    # Generate detailed analysis
                    logger.info(f"Generating detailed analysis for {entity}...")
                    detailed_report = detector.generate_detailed_anomaly_report(entity)
                    logger.info(f"Detailed analysis report generated: {detailed_report}")
                    
                    # Generate visualizations
                    logger.info(f"Generating visualizations for {entity}...")
                    detector.plot_anomalies(entity, save_plots=True)
            
            # Save anomaly detection results
            detector.save_anomaly_results()
            
            # Generate anomaly report
            report = detector.generate_anomaly_report()
            logger.info("Anomaly detection report generated")
        
        logger.info("Data processing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 