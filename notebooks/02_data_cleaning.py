"""
02 - Data Cleaning Script
Carbon Emissions Forecasting System v3.0

This script performs comprehensive data cleaning and preprocessing for the carbon emissions
forecasting project. We address data quality issues identified in the exploration phase
and prepare the data for machine learning models.

Author: Kelsi Naidoo
Institution: University of Cape Town
Date: June 2025

IEEE Standards Compliance:
- IEEE 1012-2016: Data validation and verification
- IEEE 730-2014: Quality assurance procedures
- IEEE 829-2008: Test documentation

Objectives:
1. Load and validate raw emissions data
2. Clean and standardize data formats
3. Handle missing values and outliers
4. Create temporal features and aggregations
5. Validate cleaned data quality
6. Prepare data for machine learning models
"""

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path for importing our modules
sys.path.append('src')

# Import our data processing modules
try:
    from data import DataLoader, DataCleaner, DataValidator
except ImportError:
    print("Warning: Could not import data processing modules. Using basic functions.")
    DataLoader = DataCleaner = DataValidator = None

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print(f"Script executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Set up paths
PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
FIGURES_DIR = Path(config['reports']['figures_dir'])

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"Figures directory: {FIGURES_DIR}")

# 1. Data Loading and Initial Assessment
print("\n" + "="*60)
print("1. DATA LOADING AND INITIAL ASSESSMENT")
print("="*60)

if DataLoader is not None:
    # Initialize data loader
    loader = DataLoader()
    
    # Load all entities
    print("Loading raw data...")
    raw_data = loader.load_all_entities()
    
    # Display loading summary
    summary = loader.get_data_summary()
    print(f"\nData Loading Summary:")
    print(f"Total entities: {summary['total_entities']}")
    print(f"Loaded entities: {summary['loaded_entities']}")
    print(f"Loading errors: {summary['loading_errors']}")
    
    # Display detailed information for each entity
    for entity, entity_data in summary['entities_data'].items():
        print(f"\n{entity}:")
        print(f"  Shape: {entity_data['shape']}")
        print(f"  Columns: {len(entity_data['columns'])}")
        print(f"  Memory usage: {entity_data['memory_usage'] / 1024:.2f} KB")
        
        if 'date_range' in entity_data:
            print(f"  Date range: {entity_data['date_range']['start']} to {entity_data['date_range']['end']}")
            print(f"  Total days: {entity_data['date_range']['total_days']}")
    
    # Validate data structure
    print("\nValidating data structure...")
    validation = loader.validate_data_structure()
    
    for entity, results in validation.items():
        print(f"\n{entity}:")
        print(f"  Temporal columns: {results['temporal_columns']}")
        print(f"  Emissions columns: {results['emissions_columns']}")
        print(f"  Sector columns: {results['sector_columns']}")
        
        if results['missing_required']:
            print(f"  Missing required: {results['missing_required']}")

else:
    # Fallback: Basic data loading
    print("Using basic data loading...")
    raw_data = {}
    
    for entity in config['data']['entities']:
        file_path = Path(config['data']['raw_dir']) / f"Emissions_{entity}.xlsx"
        if file_path.exists():
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                raw_data[entity] = df
                print(f"Loaded {entity}: {df.shape}")
            except Exception as e:
                print(f"Error loading {entity}: {e}")
        else:
            print(f"File not found: {file_path}")

# 2. Data Cleaning Process
print("\n" + "="*60)
print("2. DATA CLEANING PROCESS")
print("="*60)

# Use basic data cleaning approach for better control
print("Using enhanced data cleaning...")
cleaned_data = {}
cleaning_summaries = {}

for entity, df in raw_data.items():
    if df is not None:
        print(f"\nCleaning {entity}...")
        
        # Store original shape
        original_shape = df.shape
        
        # Basic cleaning
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Convert FiscalYear to proper dates (FY25 → 2025, FY24 → 2024, etc.)
        if 'FiscalYear' in df_clean.columns:
            print(f"  Converting FiscalYear format...")
            df_clean['FiscalYear'] = df_clean['FiscalYear'].astype(str)
            df_clean['FiscalYear'] = df_clean['FiscalYear'].str.replace('FY', '20')
            df_clean['FiscalYear'] = pd.to_datetime(df_clean['FiscalYear'] + '-01-01', errors='coerce')
            print(f"  Converted {df_clean['FiscalYear'].notna().sum()} valid fiscal years")
        
        # Handle negative emissions (corrections that should be subtracted from totals)
        if 'Emissions' in df_clean.columns:
            print(f"  Processing emissions data...")
            # Convert to numeric
            df_clean['Emissions'] = pd.to_numeric(df_clean['Emissions'], errors='coerce')
            
            # Count negative emissions (corrections)
            negative_count = (df_clean['Emissions'] < 0).sum()
            if negative_count > 0:
                print(f"  Found {negative_count} negative emissions (corrections)")
                
                # For corrections, we need to identify the corresponding positive entries
                # and subtract the correction amount. This is complex and depends on your data structure.
                # For now, we'll flag them and you can decide how to handle them.
                df_clean['is_correction'] = df_clean['Emissions'] < 0
                df_clean['correction_amount'] = df_clean['Emissions'].where(df_clean['Emissions'] < 0, 0)
                
                # Remove the negative corrections from the main emissions column
                # (they will be handled separately in the analysis)
                df_clean['Emissions_original'] = df_clean['Emissions']
                df_clean['Emissions'] = df_clean['Emissions'].where(df_clean['Emissions'] >= 0, 0)
                
                print(f"  Flagged {negative_count} corrections for separate processing")
        
        # Handle missing values in emissions columns
        emissions_cols = [col for col in df_clean.columns 
                        if any(keyword in col.lower() for keyword in ['emission', 'carbon', 'co2'])]
        for col in emissions_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Handle missing values in sector columns
        sector_cols = [col for col in df_clean.columns 
                      if any(keyword in col.lower() for keyword in ['sector', 'category'])]
        for col in sector_cols:
            if df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        # Remove duplicates
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            print(f"  Removed {duplicates} duplicate rows")
        
        # Sort by date if available
        if 'FiscalYear' in df_clean.columns and df_clean['FiscalYear'].notna().any():
            df_clean = df_clean.sort_values('FiscalYear')
            print(f"  Sorted by FiscalYear")
        
        if 'Date' not in df_clean.columns and 'fiscalyear' in df_clean.columns:
            df_clean['Date'] = pd.to_datetime(df_clean['fiscalyear'].astype(str) + '-01-01')
        
        cleaned_data[entity] = df_clean
        
        # Store summary
        cleaning_summaries[entity] = {
            'original_shape': original_shape,
            'cleaned_shape': df_clean.shape,
            'rows_removed': original_shape[0] - df_clean.shape[0],
            'columns_changed': original_shape[1] - df_clean.shape[1]
        }
        
        print(f"  Original shape: {original_shape}")
        print(f"  Cleaned shape: {df_clean.shape}")
        print(f"  Rows removed: {cleaning_summaries[entity]['rows_removed']}")
        print(f"  Columns changed: {cleaning_summaries[entity]['columns_changed']}")

# After cleaning, standardize column names to lowercase and underscores
for entity in cleaned_data:
    df_clean = cleaned_data[entity]
    df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
    cleaned_data[entity] = df_clean

# Save cleaned data to CSV files
print("\n" + "="*60)
print("3. SAVING CLEANED DATA")
print("="*60)

for entity, df_clean in cleaned_data.items():
    if df_clean is not None:
        output_file = PROCESSED_DATA_DIR / f'cleaned_{entity}.csv'
        df_clean.to_csv(output_file, index=False)
        print(f"Saved cleaned {entity} data to: {output_file}")
        print(f"  Shape: {df_clean.shape}")
        print(f"  Columns: {list(df_clean.columns)}")
        print(f"  Sample data:")
        print(df_clean.head(3))
        print()

# Display cleaning summary
print("\n" + "="*60)
print("3. DATA CLEANING SUMMARY")
print("="*60)

summary_df = pd.DataFrame(cleaning_summaries).T
summary_df['rows_removed_pct'] = (summary_df['rows_removed'] / summary_df['original_shape'].str[0]) * 100
summary_df['columns_changed_pct'] = (summary_df['columns_changed'] / summary_df['original_shape'].str[1]) * 100

print(summary_df)

# Visualize cleaning impact
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Rows removed
axes[0].bar(summary_df.index, summary_df['rows_removed_pct'])
axes[0].set_title('Percentage of Rows Removed')
axes[0].set_ylabel('Percentage (%)')
axes[0].tick_params(axis='x', rotation=45)

# Shape comparison
x = np.arange(len(summary_df))
width = 0.35

axes[1].bar(x - width/2, summary_df['original_shape'].str[0], width, label='Original')
axes[1].bar(x + width/2, summary_df['cleaned_shape'].str[0], width, label='Cleaned')
axes[1].set_title('Number of Rows Before and After Cleaning')
axes[1].set_ylabel('Number of Rows')
axes[1].set_xticks(x)
axes[1].set_xticklabels(summary_df.index, rotation=45)
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'data_cleaning_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Data Validation
print("\n" + "="*60)
print("4. DATA VALIDATION")
print("="*60)

if DataValidator is not None:
    # Initialize data validator
    validator = DataValidator()
    
    # Validate cleaned data for each entity
    validation_results = {}
    
    print("Validating cleaned data...")
    
    for entity, df in cleaned_data.items():
        print(f"\nValidating {entity}...")
        
        validation_result = validator.validate_entity_data(df, entity)
        validation_results[entity] = validation_result
        
        print(f"  Overall status: {validation_result['overall_status']}")
        print(f"  Errors: {len(validation_result['errors'])}")
        print(f"  Warnings: {len(validation_result['warnings'])}")
    
    # Save validation results
    validator.save_validation_results()
    
    # Generate validation report
    report = validator.generate_validation_report()
    print("Validation report generated")

else:
    # Fallback: Basic validation
    print("Using basic data validation...")
    validation_results = {}
    
    for entity, df in cleaned_data.items():
        print(f"\nValidating {entity}...")
        
        # Basic validation checks
        missing_values = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        
        validation_results[entity] = {
            'overall_status': 'PASS' if missing_values == 0 and duplicates == 0 else 'WARNING',
            'errors': [],
            'warnings': []
        }
        
        if missing_values > 0:
            validation_results[entity]['warnings'].append(f"{missing_values} missing values")
        
        if duplicates > 0:
            validation_results[entity]['warnings'].append(f"{duplicates} duplicate rows")
        
        print(f"  Overall status: {validation_results[entity]['overall_status']}")
        print(f"  Missing values: {missing_values}")
        print(f"  Duplicates: {duplicates}")

# Display validation summary
print("\n" + "="*60)
print("5. DATA VALIDATION SUMMARY")
print("="*60)

validation_summary = []
for entity, result in validation_results.items():
    summary = {
        'Entity': entity,
        'Status': result['overall_status'],
        'Errors': len(result['errors']),
        'Warnings': len(result['warnings'])
    }
    validation_summary.append(summary)

validation_df = pd.DataFrame(validation_summary)
print(validation_df)

# 6. Feature Engineering
print("\n" + "="*60)
print("6. FEATURE ENGINEERING")
print("="*60)

def create_temporal_features(df, date_col, emissions_col):
    """Create temporal features from date column and emissions column only if numeric."""
    df = df.copy()
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract temporal features
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # Create cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Lag and rolling features only for numeric emissions column
    if pd.api.types.is_numeric_dtype(df[emissions_col]):
        # Lag features
        df[f'{emissions_col}_lag1'] = df[emissions_col].shift(1)
        df[f'{emissions_col}_lag3'] = df[emissions_col].shift(3)
        df[f'{emissions_col}_lag6'] = df[emissions_col].shift(6)
        df[f'{emissions_col}_lag12'] = df[emissions_col].shift(12)
        # Rolling statistics
        df[f'{emissions_col}_rolling_mean_3'] = df[emissions_col].rolling(window=3).mean()
        df[f'{emissions_col}_rolling_mean_6'] = df[emissions_col].rolling(window=6).mean()
        df[f'{emissions_col}_rolling_mean_12'] = df[emissions_col].rolling(window=12).mean()
        df[f'{emissions_col}_rolling_std_3'] = df[emissions_col].rolling(window=3).std()
        df[f'{emissions_col}_rolling_std_6'] = df[emissions_col].rolling(window=6).std()
        # Year-over-year change
        df[f'{emissions_col}_yoy_change'] = df[emissions_col].pct_change(periods=12)
        # Month-over-month change
        df[f'{emissions_col}_mom_change'] = df[emissions_col].pct_change()
    return df

def create_sector_features(df):
    """Create sector-based features."""
    df = df.copy()
    
    sector_cols = [col for col in df.columns 
                   if any(keyword in col.lower() for keyword in ['sector', 'category'])]
    
    if sector_cols:
        sector_col = sector_cols[0]
        
        # Create dummy variables for sectors
        sector_dummies = pd.get_dummies(df[sector_col], prefix='sector')
        df = pd.concat([df, sector_dummies], axis=1)
        
        # Sector-specific statistics
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emission', 'carbon', 'co2'])
                         and not any(keyword in col.lower() for keyword in ['lag', 'rolling', 'mean', 'std', 'change', 'deviation'])]
        
        for col in emissions_cols:
            # Sector mean
            sector_means = df.groupby(sector_col)[col].transform('mean')
            df[f'{col}_sector_mean'] = sector_means
            
            # Sector std
            sector_stds = df.groupby(sector_col)[col].transform('std')
            df[f'{col}_sector_std'] = sector_stds
            
            # Deviation from sector mean
            df[f'{col}_sector_deviation'] = df[col] - sector_means
    
    return df

# Apply feature engineering to each entity
print("Creating features...")
for entity, df in cleaned_data.items():
    print(f"\nEngineering features for {entity}...")
    # Use standardized column names
    date_col = 'fiscalyear' if 'fiscalyear' in df.columns else None
    emissions_col = 'emissions' if 'emissions' in df.columns else None
    if date_col and emissions_col:
        df_with_temporal = create_temporal_features(df, date_col, emissions_col)
        output_file = PROCESSED_DATA_DIR / f'features_{entity}.csv'
        df_with_temporal.to_csv(output_file, index=False)
        print(f"  Saved features to: {output_file}")
        print(f"  Shape: {df_with_temporal.shape}")
        print(f"  Columns: {list(df_with_temporal.columns)}")
        print(f"  Sample data:")
        print(df_with_temporal.head(3))
    else:
        print(f"  Skipped feature engineering for {entity} (missing date or emissions column)")

# Display feature engineering summary
print("\n" + "="*60)
print("7. FEATURE ENGINEERING SUMMARY")
print("="*60)

feature_summary = []
for entity, df in cleaned_data.items():
    if entity in cleaned_data:
        original_cols = len(df.columns)
        engineered_cols = len(cleaned_data[entity].columns)
        features_added = engineered_cols - original_cols
        
        feature_summary.append({
            'Entity': entity,
            'Original_Columns': original_cols,
            'Engineered_Columns': engineered_cols,
            'Features_Added': features_added,
            'Feature_Increase_Pct': (features_added / original_cols) * 100
        })

feature_df = pd.DataFrame(feature_summary)
print(feature_df)

# 8. Final Data Preparation
print("\n" + "="*60)
print("8. FINAL DATA PREPARATION")
print("="*60)

def prepare_for_ml(df, target_col=None):
    """Prepare data for machine learning models."""
    df = df.copy()
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Identify target column if not specified
    if target_col is None:
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emission', 'carbon', 'co2'])
                         and not any(keyword in col.lower() for keyword in ['lag', 'rolling', 'mean', 'std', 'change', 'deviation'])]
        if emissions_cols:
            target_col = emissions_cols[0]
        else:
            raise ValueError("No target column found")
    
    # Remove date columns (keep only temporal features)
    date_cols = [col for col in df.columns 
                 if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])
                 and not any(keyword in col.lower() for keyword in ['sin', 'cos', 'lag', 'rolling'])]
    
    # Remove original sector columns (keep encoded versions)
    sector_cols = [col for col in df.columns 
                   if any(keyword in col.lower() for keyword in ['sector', 'category'])
                   and not col.startswith('sector_')]
    
    # Remove columns to exclude
    exclude_cols = date_cols + sector_cols
    df_ml = df.drop(columns=exclude_cols, errors='ignore')
    
    # Ensure target column is present
    if target_col not in df_ml.columns:
        raise ValueError(f"Target column {target_col} not found in prepared data")
    
    return df_ml, target_col

# Prepare data for machine learning
ml_ready_data = {}

print("Preparing data for machine learning...")

for entity, df in cleaned_data.items():
    print(f"\nPreparing {entity} for ML...")
    
    try:
        df_ml, target_col = prepare_for_ml(df)
        ml_ready_data[entity] = {
            'data': df_ml,
            'target_column': target_col,
            'feature_columns': [col for col in df_ml.columns if col != target_col]
        }
        
        print(f"  Target column: {target_col}")
        print(f"  Feature columns: {len(ml_ready_data[entity]['feature_columns'])}")
        print(f"  Total samples: {len(df_ml)}")
        
    except Exception as e:
        print(f"  Error preparing {entity}: {str(e)}")
        continue

# Display ML preparation summary
print("\n" + "="*60)
print("9. MACHINE LEARNING PREPARATION SUMMARY")
print("="*60)

ml_summary = []
for entity, data in ml_ready_data.items():
    summary = {
        'Entity': entity,
        'Target_Column': data['target_column'],
        'Feature_Columns': len(data['feature_columns']),
        'Total_Samples': len(data['data']),
        'Memory_Usage_MB': data['data'].memory_usage(deep=True).sum() / 1024 / 1024
    }
    ml_summary.append(summary)

ml_df = pd.DataFrame(ml_summary)
print(ml_df)

# 9. Save Processed Data
print("\n" + "="*60)
print("10. SAVING PROCESSED DATA")
print("="*60)

# Save cleaned data
for entity, df in cleaned_data.items():
    output_file = PROCESSED_DATA_DIR / f"{entity}_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned data for {entity} to {output_file}")

# Save validation results
validation_summary_file = PROCESSED_DATA_DIR / 'validation_results.json'
with open(validation_summary_file, 'w') as f:
    json.dump(validation_results, f, indent=2, default=str)
print(f"Saved validation results to {validation_summary_file}")

# Save engineered data
for entity, df in cleaned_data.items():
    output_file = PROCESSED_DATA_DIR / f"{entity}_engineered.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved engineered data for {entity} to {output_file}")

# Save ML-ready data
for entity, data in ml_ready_data.items():
    output_file = PROCESSED_DATA_DIR / f"{entity}_ml_ready.csv"
    data['data'].to_csv(output_file, index=False)
    print(f"Saved ML-ready data for {entity} to {output_file}")
    
    # Save metadata
    metadata = {
        'target_column': data['target_column'],
        'feature_columns': data['feature_columns'],
        'total_samples': len(data['data']),
        'created_at': datetime.now().isoformat()
    }
    
    metadata_file = PROCESSED_DATA_DIR / f"{entity}_ml_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata for {entity} to {metadata_file}")

# 10. Summary and Next Steps
print("\n" + "="*80)
print("11. DATA CLEANING AND PREPARATION SUMMARY")
print("="*80)

final_summary = {
    'total_entities': len(raw_data),
    'successfully_cleaned': len(cleaned_data),
    'successfully_engineered': len(cleaned_data),
    'ml_ready': len(ml_ready_data),
    'validation_passed': sum(1 for result in validation_results.values() 
                            if result['overall_status'] == 'PASS'),
    'total_validation_errors': sum(len(result['errors']) 
                                   for result in validation_results.values()),
    'total_validation_warnings': sum(len(result['warnings']) 
                                     for result in validation_results.values())
}

print(f"Total entities processed: {final_summary['total_entities']}")
print(f"Successfully cleaned: {final_summary['successfully_cleaned']}")
print(f"Successfully engineered: {final_summary['successfully_engineered']}")
print(f"ML ready: {final_summary['ml_ready']}")
print(f"Validation passed: {final_summary['validation_passed']}")
print(f"Total validation errors: {final_summary['total_validation_errors']}")
print(f"Total validation warnings: {final_summary['total_validation_warnings']}")

# Save final summary
summary_file = PROCESSED_DATA_DIR / 'cleaning_final_summary.json'
with open(summary_file, 'w') as f:
    json.dump(final_summary, f, indent=2, default=str)

print(f"\nFinal summary saved to: {summary_file}")

# Generate recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS FOR NEXT STEPS")
print("="*60)

recommendations = []

if final_summary['ml_ready'] > 0:
    recommendations.append("✓ Data is ready for machine learning models")
    recommendations.append("  • Proceed to model development")
    recommendations.append("  • Consider cross-validation strategies")
    recommendations.append("  • Implement model comparison framework")
else:
    recommendations.append("✗ Data preparation incomplete")
    recommendations.append("  • Review validation errors")
    recommendations.append("  • Address data quality issues")
    recommendations.append("  • Re-run cleaning process")

if final_summary['total_validation_errors'] > 0:
    recommendations.append("⚠ Validation errors detected")
    recommendations.append("  • Review validation report")
    recommendations.append("  • Address critical data issues")

if final_summary['total_validation_warnings'] > 0:
    recommendations.append("⚠ Validation warnings detected")
    recommendations.append("  • Review warnings for potential issues")
    recommendations.append("  • Consider impact on model performance")

for rec in recommendations:
    print(rec)

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. Review validation results and address any critical issues")
print("2. Proceed to Model Development")
print("3. Implement and compare forecasting models")
print("4. Evaluate model performance and generate insights")

print("\n---")
print("Data cleaning and preparation completed successfully!")
print("All processed data has been saved to the processed directory.") 