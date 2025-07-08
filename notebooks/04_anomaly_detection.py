"""
Anomaly Detection for Carbon Emissions Data
===========================================

This notebook implements comprehensive anomaly detection for carbon emissions data
using multiple algorithms and provides detailed analysis and visualization.

Author: Kelsi Naidoo
Institution: University of Cape Town
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Import project modules
from src.models.anomaly_detector import AnomalyDetector
from src.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("ANOMALY DETECTION FOR CARBON EMISSIONS DATA")
print("=" * 80)
print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load configuration
config_path = project_root / "config" / "config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

print(f"Project: {config['project']['name']} v{config['project']['version']}")
print(f"Entities: {config['data']['entities']}")
print()

# Initialize anomaly detector
print("Initializing anomaly detector...")
detector = AnomalyDetector(str(config_path))
print("✓ Anomaly detector initialized")
print()

# Load cleaned data
print("Loading cleaned data...")
cleaned_data = detector.load_cleaned_data()

for entity, df in cleaned_data.items():
    if df is not None:
        print(f"✓ Loaded {entity}: {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        print(f"✗ No data available for {entity}")
print()

# Perform anomaly detection
print("=" * 50)
print("PERFORMING ANOMALY DETECTION")
print("=" * 50)

anomaly_results = {}

for entity, df in cleaned_data.items():
    if df is not None:
        print(f"\nAnalyzing {entity}...")
        print("-" * 30)
        
        # Detect anomalies
        result = detector.detect_anomalies(df, entity)
        anomaly_results[entity] = result
        
        # Print summary
        print(f"Total samples: {result['total_samples']}")
        print(f"Features used: {len(result['feature_columns'])}")
        print("\nAnomaly detection results:")
        
        for model_name, model_result in result['anomalies'].items():
            if 'error' in model_result:
                print(f"  {model_name}: ERROR - {model_result['error']}")
            else:
                print(f"  {model_name}: {model_result['count']} anomalies ({model_result['percentage']:.2f}%)")
        
        print()

# Generate and save results
print("=" * 50)
print("SAVING RESULTS")
print("=" * 50)

detector.save_anomaly_results()
report_path = detector.generate_anomaly_report()

print(f"✓ Results saved to reports directory")
print(f"✓ Report generated: {report_path}")
print()

# Generate visualizations
print("=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

for entity in anomaly_results.keys():
    print(f"\nGenerating plots for {entity}...")
    detector.plot_anomalies(entity, save_plots=True)
    print(f"✓ Plots saved for {entity}")

print("\n" + "=" * 50)
print("DETAILED ANOMALY ANALYSIS")
print("=" * 50)

# Generate detailed analysis for each entity
for entity, result in anomaly_results.items():
    print(f"\n{entity.upper()} - DETAILED ANALYSIS")
    print("=" * 40)
    
    # Generate detailed anomaly report
    print(f"Generating detailed analysis report for {entity}...")
    detailed_report_path = detector.generate_detailed_anomaly_report(entity)
    print(f"✓ Detailed report generated: {detailed_report_path}")
    
    # Perform detailed analysis
    analysis = detector.analyze_anomaly_details(entity)
    
    if analysis:
        print(f"\nSummary for {entity}:")
        print(f"  Total samples: {analysis['total_samples']}")
        print(f"  Models with anomalies: {len(analysis['anomaly_details'])}")
        
        # Show consensus anomalies
        if analysis['consensus_analysis']:
            consensus = analysis['consensus_analysis']
            print(f"  Consensus anomalies: {consensus['count']} ({consensus['percentage']:.2f}%)")
            
            # Show consensus sources if available
            if 'sources' in consensus:
                print(f"  Consensus anomalies by source:")
                for source, count in consensus['sources'].items():
                    print(f"    {source}: {count}")
            
            # Show consensus temporal if available
            if 'temporal' in consensus and consensus['temporal']['years']:
                print(f"  Consensus anomalies by year:")
                for year, count in sorted(consensus['temporal']['years'].items()):
                    print(f"    {year}: {count}")
        
        # Show source analysis
        if analysis['source_analysis']:
            print(f"\n  Anomalies by emission source:")
            for model_name, source_counts in analysis['source_analysis'].items():
                print(f"    {model_name}:")
                for source, count in source_counts.items():
                    print(f"      {source}: {count}")
    
    print()

print("\n" + "=" * 50)
print("DETAILED ANALYSIS")
print("=" * 50)

# Cross-entity comparison
print("\n" + "=" * 50)
print("CROSS-ENTITY COMPARISON")
print("=" * 50)

comparison_data = []
for entity, result in anomaly_results.items():
    for model_name, model_result in result['anomalies'].items():
        if 'error' not in model_result:
            comparison_data.append({
                'entity': entity,
                'model': model_name,
                'anomaly_count': model_result['count'],
                'anomaly_percentage': model_result['percentage'],
                'total_samples': result['total_samples']
            })

comparison_df = pd.DataFrame(comparison_data)

if not comparison_df.empty:
    print("\nAnomaly Detection Summary by Entity and Model:")
    print(comparison_df.to_string(index=False))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Pivot for plotting
    pivot_df = comparison_df.pivot(index='entity', columns='model', values='anomaly_percentage')
    
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Anomaly Detection Results by Entity and Model')
    plt.xlabel('Entity')
    plt.ylabel('Anomaly Percentage (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_plot_path = project_root / "reports" / f"anomaly_comparison_{timestamp}.png"
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved: {comparison_plot_path}")
    
    plt.show()

# Recommendations
print("\n" + "=" * 50)
print("RECOMMENDATIONS")
print("=" * 50)

print("\nBased on the anomaly detection results, consider the following actions:")
print()
print("1. HIGH PRIORITY:")
print("   - Investigate data points flagged by multiple algorithms")
print("   - Review periods with unusually high anomaly rates")
print("   - Validate data quality during high-anomaly periods")
print()
print("2. MEDIUM PRIORITY:")
print("   - Analyze feature importance for anomaly detection")
print("   - Consider domain-specific anomaly thresholds")
print("   - Review seasonal patterns in anomaly detection")
print()
print("3. LOW PRIORITY:")
print("   - Fine-tune algorithm parameters based on domain knowledge")
print("   - Implement real-time anomaly monitoring")
print("   - Create automated alerts for new anomalies")

print("\n" + "=" * 80)
print("ANOMALY DETECTION COMPLETED")
print("=" * 80)
print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Check the reports directory for detailed results and visualizations.") 