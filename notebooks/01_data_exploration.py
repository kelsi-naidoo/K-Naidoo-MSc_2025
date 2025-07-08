"""
01 - Data Exploration Script
Carbon Emissions Forecasting System v3.0

This script performs initial data exploration for the carbon emissions forecasting project.
We examine the structure, quality, and characteristics of the emissions data from EntityA and EntityB.

Author: Kelsi Naidoo
Institution: University of Cape Town
Date: June 2025

IEEE Standards Compliance:
- IEEE 1012-2016: Data validation and verification
- IEEE 829-2008: Test documentation
- IEEE 730-2014: Quality assurance procedures

Objectives:
1. Load and examine raw emissions data
2. Understand data structure and schema
3. Identify data quality issues
4. Explore temporal patterns and seasonality
5. Analyze sector-based emissions distribution
6. Document findings for downstream processing
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
RAW_DATA_DIR = Path(config['data']['raw_dir'])
PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
REPORTS_DIR = Path(config['reports']['output_dir'])
FIGURES_DIR = Path(config['reports']['figures_dir'])

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"Reports directory: {REPORTS_DIR}")

def load_entity_data(entity_name):
    """Load emissions data for a specific entity."""
    file_pattern = config['data']['file_patterns']['emissions']
    file_path = RAW_DATA_DIR / file_pattern.replace('*', entity_name)
    
    if not file_path.exists():
        print(f"Warning: File not found for {entity_name}: {file_path}")
        return None
    
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Successfully loaded {entity_name} data: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {entity_name} data: {e}")
        return None

# Load data for both entities
print("\n" + "="*60)
print("1. DATA LOADING AND INITIAL INSPECTION")
print("="*60)

entities_data = {}
for entity in config['data']['entities']:
    entities_data[entity] = load_entity_data(entity)

# Display basic information for each entity
for entity, df in entities_data.items():
    if df is not None:
        print(f"\n{'='*50}")
        print(f"ENTITY: {entity}")
        print(f"{'='*50}")
        
        print(f"\nShape: {df.shape}")
        print(f"\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nData Types:")
        print(df.dtypes)
        
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nLast 5 rows:")
        print(df.tail())

def assess_data_quality(df, entity_name):
    """Comprehensive data quality assessment."""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY ASSESSMENT: {entity_name}")
    print(f"{'='*60}")
    
    # Missing values
    print(f"\n1. MISSING VALUES:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✓ No missing values found")
    
    # Duplicate rows
    print(f"\n2. DUPLICATE ROWS:")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data types and unique values
    print(f"\n3. COLUMN ANALYSIS:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {df[col].dtype} | {unique_count} unique values")
        
        # Show sample values for categorical columns
        if df[col].dtype == 'object' and unique_count < 20:
            print(f"    Sample values: {df[col].unique()[:5]}")
    
    return missing_df

# Assess quality for each entity
print("\n" + "="*60)
print("2. DATA QUALITY ASSESSMENT")
print("="*60)

quality_reports = {}
for entity, df in entities_data.items():
    if df is not None:
        quality_reports[entity] = assess_data_quality(df, entity)

def analyze_temporal_patterns(df, entity_name):
    """Analyze temporal patterns in the data."""
    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS: {entity_name}")
    print(f"{'='*60}")
    
    # Identify date columns
    date_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time']):
            date_columns.append(col)
    
    print(f"\nDate columns found: {date_columns}")
    
    if date_columns:
        # Analyze each date column
        for date_col in date_columns:
            print(f"\nAnalyzing {date_col}:")
            
            # Convert to datetime if possible
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates = df[date_col].notna().sum()
                print(f"  Valid dates: {valid_dates}/{len(df)} ({valid_dates/len(df)*100:.1f}%)")
                
                if valid_dates > 0:
                    print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
                    print(f"  Total days: {(df[date_col].max() - df[date_col].min()).days}")
                    
                    # Monthly distribution
                    monthly_counts = df[date_col].dt.month.value_counts().sort_index()
                    print(f"  Monthly distribution: {dict(monthly_counts)}")
                    
            except Exception as e:
                print(f"  Error processing {date_col}: {e}")
    else:
        print("\nNo date columns identified")

# Analyze temporal patterns for each entity
print("\n" + "="*60)
print("3. TEMPORAL ANALYSIS")
print("="*60)

for entity, df in entities_data.items():
    if df is not None:
        analyze_temporal_patterns(df, entity)

def analyze_sectors(df, entity_name):
    """Analyze sector-based emissions distribution."""
    print(f"\n{'='*60}")
    print(f"SECTOR ANALYSIS: {entity_name}")
    print(f"{'='*60}")
    
    # For EntityB, use 'Property Type' as sector
    if entity_name == 'EntityB' and 'Property Type' in df.columns:
        sector_col = 'Property Type'
    else:
        # For others, use 'Sector' or fallback to first sector-like column
        sector_candidates = [col for col in df.columns if 'sector' in col.lower() or col == 'Sector']
        sector_col = sector_candidates[0] if sector_candidates else None
    
    if sector_col:
        print(f"\nSector column: {sector_col}")
        
        # Sector distribution
        sector_counts = df[sector_col].value_counts()
        print(f"\nSector distribution:")
        for sector, count in sector_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sector}: {count} records ({percentage:.1f}%)")
        
        # Visualize sector distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sector_counts.plot(kind='bar')
        plt.title(f'Sector Distribution - {entity_name}')
        plt.xlabel('Sector')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        plt.pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
        plt.title(f'Sector Distribution - {entity_name}')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'{entity_name}_sector_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("\nNo sector column found")

# Analyze sectors for each entity
print("\n" + "="*60)
print("4. SECTOR ANALYSIS")
print("="*60)

for entity, df in entities_data.items():
    if df is not None:
        analyze_sectors(df, entity)

def analyze_emissions(df, entity_name):
    """Analyze emissions data characteristics."""
    print(f"\n{'='*60}")
    print(f"EMISSIONS ANALYSIS: {entity_name}")
    print(f"{'='*60}")
    
    # Find emissions columns - look for actual column names
    emissions_columns = []
    for col in df.columns:
        if 'emission' in col.lower() or col == 'Emissions' or col == 'Emissions Source':
            emissions_columns.append(col)
    
    if emissions_columns:
        print(f"\nEmissions columns found: {emissions_columns}")
        
        for em_col in emissions_columns:
            print(f"\nAnalyzing {em_col}:")
            
            # For 'Emissions Source', treat as categorical
            if em_col == 'Emissions Source':
                print(f"  Statistics:")
                print(f"    Count: {df[em_col].count()}")
                print(f"    Unique values: {df[em_col].nunique()}")
                print(f"    Missing values: {df[em_col].isnull().sum()}")
                
                # Distribution of emissions sources
                source_counts = df[em_col].value_counts()
                print(f"  Top 10 Emissions Sources:")
                for source, count in source_counts.head(10).items():
                    print(f"    {source}: {count}")
                
                # Visualize emissions source distribution
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                source_counts.head(10).plot(kind='bar')
                plt.title(f'Top 10 Emissions Sources - {entity_name}')
                plt.xlabel('Emissions Source')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 3, 2)
                plt.boxplot([df[df[em_col] == source]['Emissions'].values 
                           for source in source_counts.head(5).index])
                plt.title(f'Emissions by Top 5 Sources - {entity_name}')
                plt.xlabel('Emissions Source')
                plt.ylabel('Emissions')
                plt.xticks(range(1, 6), source_counts.head(5).index, rotation=45)
                
                plt.subplot(1, 3, 3)
                from scipy import stats
                emissions_data = df['Emissions'].dropna()
                if len(emissions_data) > 0:
                    stats.probplot(emissions_data, dist="norm", plot=plt)
                    plt.title(f'Emissions Q-Q Plot - {entity_name}')
                
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f'{entity_name}_Emissions_Source_distribution.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            else:
                # For 'Emissions' column, treat as numeric
                # Convert to numeric
                df[em_col] = pd.to_numeric(df[em_col], errors='coerce')
                
                # Basic statistics
                stats = df[em_col].describe()
                print(f"  Statistics:")
                print(f"    Count: {stats['count']:.0f}")
                print(f"    Mean: {stats['mean']:.2f}")
                print(f"    Std: {stats['std']:.2f}")
                print(f"    Min: {stats['min']:.2f}")
                print(f"    25%: {stats['25%']:.2f}")
                print(f"    50%: {stats['50%']:.2f}")
                print(f"    75%: {stats['75%']:.2f}")
                print(f"    Max: {stats['max']:.2f}")
                
                # Missing values
                missing = df[em_col].isnull().sum()
                print(f"  Missing values: {missing} ({missing/len(df)*100:.1f}%)")
                
                # Distribution plot
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.hist(df[em_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{em_col} Distribution - {entity_name}')
                plt.xlabel('Emissions')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 3, 2)
                plt.boxplot(df[em_col].dropna())
                plt.title(f'{em_col} Box Plot - {entity_name}')
                plt.ylabel('Emissions')
                
                plt.subplot(1, 3, 3)
                from scipy import stats
                stats.probplot(df[em_col].dropna(), dist="norm", plot=plt)
                plt.title(f'{em_col} Q-Q Plot - {entity_name}')
                
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f'{entity_name}_{em_col}_distribution.png', dpi=300, bbox_inches='tight')
                plt.show()
            
    else:
        print("\nNo emissions columns found")

# Analyze emissions for each entity
print("\n" + "="*60)
print("5. EMISSIONS DATA ANALYSIS")
print("="*60)

for entity, df in entities_data.items():
    if df is not None:
        analyze_emissions(df, entity)

# Generate summary report
print("\n" + "="*80)
print("6. DATA EXPLORATION SUMMARY REPORT")
print("="*80)

summary_data = []

for entity, df in entities_data.items():
    if df is not None:
        # Basic info
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_total = df.isnull().sum().sum()
        missing_percent = (missing_total / (total_rows * total_cols)) * 100
        
        # Date columns
        date_cols = [col for col in df.columns 
                    if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
        
        # Sector columns
        sector_cols = [col for col in df.columns 
                      if any(keyword in col.lower() for keyword in ['sector', 'category'])]
        
        # Emissions columns
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emission', 'carbon', 'co2'])]
        
        summary_data.append({
            'Entity': entity,
            'Total_Rows': total_rows,
            'Total_Columns': total_cols,
            'Missing_Values': missing_total,
            'Missing_Percent': f"{missing_percent:.1f}%",
            'Date_Columns': len(date_cols),
            'Sector_Columns': len(sector_cols),
            'Emissions_Columns': len(emissions_cols)
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df)

# Save summary
summary_df.to_csv(PROCESSED_DATA_DIR / 'data_exploration_summary.csv', index=False)
print(f"\nSummary saved to: {PROCESSED_DATA_DIR / 'data_exploration_summary.csv'}")

# Generate recommendations
print("\n" + "="*80)
print("7. RECOMMENDATIONS FOR DATA CLEANING")
print("="*80)

recommendations = []

for entity, df in entities_data.items():
    if df is not None:
        print(f"\n{entity}:")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"  • Missing values found in: {missing_cols}")
            recommendations.append(f"{entity}: Handle missing values in {missing_cols}")
        
        # Check for date columns
        date_cols = [col for col in df.columns 
                    if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
        if date_cols:
            print(f"  • Date columns: {date_cols}")
            recommendations.append(f"{entity}: Standardize date format in {date_cols}")
        else:
            print(f"  • No date columns identified")
            recommendations.append(f"{entity}: Identify or create temporal index")
        
        # Check for emissions columns
        emissions_cols = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['emission', 'carbon', 'co2'])]
        if emissions_cols:
            print(f"  • Emissions columns: {emissions_cols}")
            recommendations.append(f"{entity}: Validate emissions data in {emissions_cols}")
        else:
            print(f"  • No emissions columns found")
            recommendations.append(f"{entity}: Identify target variable for forecasting")
        
        # Check for sector columns
        sector_cols = [col for col in df.columns 
                      if any(keyword in col.lower() for keyword in ['sector', 'category'])]
        if sector_cols:
            print(f"  • Sector columns: {sector_cols}")
            recommendations.append(f"{entity}: Standardize sector categories in {sector_cols}")

# Save recommendations
with open(PROCESSED_DATA_DIR / 'cleaning_recommendations.txt', 'w') as f:
    f.write("Data Cleaning Recommendations\n")
    f.write("="*50 + "\n\n")
    for rec in recommendations:
        f.write(f"• {rec}\n")

print(f"\nRecommendations saved to: {PROCESSED_DATA_DIR / 'cleaning_recommendations.txt'}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("1. Review validation results and address any critical issues")
print("2. Proceed to Script 02: Data Cleaning")
print("3. Implement and compare forecasting models")
print("4. Evaluate model performance and generate insights")

print("\n---")
print("Data exploration completed successfully!")
print("All findings have been documented and saved for reproducibility.") 