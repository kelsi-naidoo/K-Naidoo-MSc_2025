"""
Anomaly Detection Module for Carbon Emissions Forecasting System v3.0.

This module implements various anomaly detection algorithms for
identifying unusual patterns in carbon emissions data.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest as PyODIForest
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection system for carbon emissions data.
    
    Implements multiple anomaly detection algorithms:
    - Isolation Forest
    - Local Outlier Factor (LOF)
    - Elliptic Envelope
    - Cluster-Based Local Outlier Factor (CBLOF)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the anomaly detector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        self.models = {}
        
        # Create output directories
        self.output_dir = Path(self.config['data']['processed_dir'])
        self.reports_dir = Path(self.config['reports']['output_dir'])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        logger.info("AnomalyDetector initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def load_cleaned_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load cleaned data for anomaly detection.
        
        Returns:
            Dictionary of cleaned dataframes for each entity
        """
        cleaned_data = {}
        processed_dir = Path(self.config['data']['processed_dir'])
        
        for entity in self.config['data']['entities']:
            entity_name = entity
            file_path = processed_dir / f"cleaned_{entity_name}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Add Date column from fiscalyear if it doesn't exist
                    if 'Date' not in df.columns and 'fiscalyear' in df.columns:
                        df['Date'] = pd.to_datetime(df['fiscalyear'].astype(str) + '-01-01')
                    df['Date'] = pd.to_datetime(df['Date'])
                    cleaned_data[entity_name] = df
                    logger.info(f"Loaded cleaned data for {entity_name}: {df.shape}")
                except Exception as e:
                    logger.error(f"Error loading cleaned data for {entity_name}: {e}")
                    cleaned_data[entity_name] = None
            else:
                logger.warning(f"Cleaned data file not found for {entity_name}: {file_path}")
                cleaned_data[entity_name] = None
        
        return cleaned_data
    
    def prepare_features(self, df: pd.DataFrame, entity: str) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Prepare features for anomaly detection.
        
        Args:
            df: Input dataframe
            entity: Entity name
            
        Returns:
            Tuple of (features array, feature dataframe)
        """
        # Select numerical columns for anomaly detection
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove date-related columns if they exist
        date_cols = [col for col in numerical_cols if 'year' in col.lower() or 'month' in col.lower()]
        numerical_cols = [col for col in numerical_cols if col not in date_cols]
        
        # Focus on emissions-specific features
        emissions_cols = [col for col in numerical_cols if 'emissions' in col.lower()]
        other_numerical = [col for col in numerical_cols if 'emissions' not in col.lower()]
        
        # Add time-based features
        feature_df = df.copy()
        feature_df['year'] = df['Date'].dt.year
        feature_df['month'] = df['Date'].dt.month
        feature_df['quarter'] = df['Date'].dt.quarter
        
        # Add emissions-specific features
        if 'emissions' in df.columns:
            feature_df['emissions_log'] = np.log1p(df['emissions'].abs())  # Log transform for skewed data
            feature_df['emissions_sqrt'] = np.sqrt(df['emissions'].abs())  # Square root transform
            feature_df['emissions_zscore'] = (df['emissions'] - df['emissions'].mean()) / df['emissions'].std()
            
            # Add lag features for time series (only if we have enough data)
            if len(df) > 12:
                feature_df['emissions_lag1'] = df['emissions'].shift(1)
                feature_df['emissions_lag3'] = df['emissions'].shift(3)
                feature_df['emissions_rolling_mean'] = df['emissions'].rolling(window=3, min_periods=1).mean()
                feature_df['emissions_rolling_std'] = df['emissions'].rolling(window=3, min_periods=1).std()
        
        # Select final feature columns - prioritize emissions features
        feature_cols = emissions_cols + ['year', 'month', 'quarter']
        if 'emissions_log' in feature_df.columns:
            feature_cols.append('emissions_log')
        if 'emissions_sqrt' in feature_df.columns:
            feature_cols.append('emissions_sqrt')
        if 'emissions_zscore' in feature_df.columns:
            feature_cols.append('emissions_zscore')
        
        # Add lag and rolling features if they exist
        lag_cols = [col for col in feature_df.columns if 'lag' in col or 'rolling' in col]
        feature_cols.extend(lag_cols)
        
        # Add a few other numerical features if they exist
        feature_cols.extend(other_numerical[:3])  # Limit to first 3 to avoid too many features
        
        # Remove any remaining NaN values
        feature_df = feature_df.dropna(subset=feature_cols)
        
        # Extract features
        features = feature_df[feature_cols].values
        
        logger.info(f"Prepared {len(features)} samples with {len(feature_cols)} features for {entity}")
        
        return features, feature_df[feature_cols]
    
    def detect_anomalies(self, df: pd.DataFrame, entity: str) -> Dict[str, Any]:
        """
        Detect anomalies using multiple algorithms.
        
        Args:
            df: Input dataframe
            entity: Entity name
            
        Returns:
            Dictionary containing anomaly detection results
        """
        if df is None or df.empty:
            logger.warning(f"No data available for anomaly detection in {entity}")
            return {}
        
        logger.info(f"Starting anomaly detection for {entity}")
        
        # Prepare features
        features, feature_df = self.prepare_features(df, entity)
        
        if len(features) == 0:
            logger.warning(f"No features available for anomaly detection in {entity}")
            return {}
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize models with more realistic contamination levels for carbon emissions data
        models = {
            'isolation_forest': IsolationForest(contamination=0.001, random_state=42, n_estimators=100),
            'lof': LocalOutlierFactor(contamination=0.001, n_neighbors=20, metric='manhattan'),
            'elliptic_envelope': EllipticEnvelope(contamination=0.001, random_state=42, support_fraction=0.8),
            'pyod_iforest': PyODIForest(contamination=0.001, random_state=42, n_estimators=100),
            'pyod_lof': LOF(contamination=0.001, n_neighbors=20),
            'cblof': CBLOF(contamination=0.001, random_state=42, n_clusters=8)
        }
        
        results = {
            'entity': entity,
            'total_samples': len(features),
            'feature_columns': feature_df.columns.tolist(),
            'anomalies': {},
            'summary': {}
        }
        
        # Run each model
        for model_name, model in models.items():
            try:
                logger.info(f"Running {model_name} for {entity}")
                
                if model_name == 'lof':
                    # LOF doesn't have fit_predict, use fit and predict separately
                    model.fit(features_scaled)
                    predictions = model.fit_predict(features_scaled)
                else:
                    predictions = model.fit_predict(features_scaled)
                
                # Convert predictions: -1 for anomalies, 1 for normal
                anomaly_indices = np.where(predictions == -1)[0]
                anomaly_scores = model.decision_function(features_scaled) if hasattr(model, 'decision_function') else None
                
                results['anomalies'][model_name] = {
                    'indices': anomaly_indices.tolist(),
                    'count': len(anomaly_indices),
                    'percentage': len(anomaly_indices) / len(features) * 100,
                    'scores': anomaly_scores.tolist() if anomaly_scores is not None else None
                }
                
                logger.info(f"{model_name}: Found {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(features)*100:.2f}%)")
                
            except Exception as e:
                logger.error(f"Error running {model_name} for {entity}: {e}")
                results['anomalies'][model_name] = {
                    'indices': [],
                    'count': 0,
                    'percentage': 0,
                    'scores': None,
                    'error': str(e)
                }
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results['anomalies'])
        
        # Store results
        self.results[entity] = results
        self.models[entity] = models
        
        return results
    
    def _generate_summary(self, anomalies: Dict) -> Dict:
        """Generate summary statistics for anomaly detection results."""
        summary = {
            'total_models': len(anomalies),
            'models_with_anomalies': 0,
            'average_anomaly_percentage': 0,
            'consensus_anomalies': 0
        }
        
        anomaly_counts = []
        for model_name, result in anomalies.items():
            if result['count'] > 0:
                summary['models_with_anomalies'] += 1
                anomaly_counts.append(result['count'])
        
        if anomaly_counts:
            summary['average_anomaly_percentage'] = np.mean(anomaly_counts) / len(anomalies)
        
        return summary
    
    def save_anomaly_results(self):
        """Save anomaly detection results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.reports_dir / f"anomaly_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for entity, result in self.results.items():
            for model_name, model_result in result['anomalies'].items():
                summary_data.append({
                    'entity': entity,
                    'model': model_name,
                    'anomaly_count': model_result['count'],
                    'anomaly_percentage': model_result['percentage'],
                    'total_samples': result['total_samples']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.reports_dir / f"anomaly_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        logger.info(f"Anomaly results saved to {results_file}")
        logger.info(f"Anomaly summary saved to {summary_file}")
    
    def generate_anomaly_report(self) -> str:
        """Generate a comprehensive anomaly detection report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"anomaly_report_{timestamp}.md"
        
        report_content = f"""# Anomaly Detection Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This report summarizes anomaly detection results for carbon emissions data across multiple entities and algorithms.

## Summary Statistics
"""
        
        # Add summary statistics
        total_entities = len(self.results)
        total_anomalies = sum(
            sum(model_result['count'] for model_result in entity_result['anomalies'].values())
            for entity_result in self.results.values()
        )
        
        report_content += f"""
- **Total Entities Analyzed**: {total_entities}
- **Total Anomalies Detected**: {total_anomalies}
- **Algorithms Used**: 6 (Isolation Forest, LOF, Elliptic Envelope, PyOD IForest, PyOD LOF, CBLOF)

## Detailed Results by Entity
"""
        
        # Add detailed results for each entity
        for entity, result in self.results.items():
            report_content += f"""
### {entity}
- **Total Samples**: {result['total_samples']}
- **Feature Columns**: {len(result['feature_columns'])}

#### Anomaly Detection Results:
"""
            
            for model_name, model_result in result['anomalies'].items():
                if 'error' in model_result:
                    report_content += f"- **{model_name}**: Error - {model_result['error']}\n"
                else:
                    report_content += f"- **{model_name}**: {model_result['count']} anomalies ({model_result['percentage']:.2f}%)\n"
        
        # Add recommendations
        report_content += """
## Recommendations

1. **Investigate High-Anomaly Periods**: Focus on time periods where multiple algorithms detect anomalies
2. **Data Quality Review**: Examine samples flagged by multiple algorithms for potential data quality issues
3. **Domain Expert Validation**: Have domain experts review detected anomalies to determine if they represent actual issues
4. **Model Tuning**: Consider adjusting contamination parameters based on domain knowledge
5. **Feature Engineering**: Explore additional features that might improve anomaly detection accuracy

## Technical Details

### Algorithms Used:
- **Isolation Forest**: Tree-based anomaly detection
- **Local Outlier Factor (LOF)**: Density-based anomaly detection
- **Elliptic Envelope**: Gaussian distribution-based anomaly detection
- **PyOD IForest**: Enhanced isolation forest implementation
- **PyOD LOF**: Enhanced LOF implementation
- **CBLOF**: Cluster-based local outlier factor

### Feature Engineering:
- Original numerical features
- Time-based features (year, month, quarter)
- Lag features (1-period and 3-period lags)
- Rolling statistics (mean, standard deviation)
"""
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Anomaly detection report generated: {report_file}")
        return str(report_file)
    
    def plot_anomalies(self, entity: str, save_plots: bool = True):
        """
        Generate visualization plots for anomaly detection results.
        
        Args:
            entity: Entity name to plot
            save_plots: Whether to save plots to files
        """
        if entity not in self.results:
            logger.warning(f"No results available for {entity}")
            return
        
        result = self.results[entity]
        
        # Load original data for plotting
        processed_dir = Path(self.config['data']['processed_dir'])
        df = pd.read_csv(processed_dir / f"cleaned_{entity}.csv")
        # Ensure Date column exists
        if 'Date' not in df.columns and 'fiscalyear' in df.columns:
            df['Date'] = pd.to_datetime(df['fiscalyear'].astype(str) + '-01-01')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Anomaly Detection Results for {entity}', fontsize=16)
        
        # Plot 1: Anomaly counts by model
        model_names = list(result['anomalies'].keys())
        anomaly_counts = [result['anomalies'][model]['count'] for model in model_names]
        
        axes[0, 0].bar(model_names, anomaly_counts)
        axes[0, 0].set_title('Anomaly Counts by Model')
        axes[0, 0].set_ylabel('Number of Anomalies')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Anomaly percentages by model
        anomaly_percentages = [result['anomalies'][model]['percentage'] for model in model_names]
        
        axes[0, 1].bar(model_names, anomaly_percentages)
        axes[0, 1].set_title('Anomaly Percentages by Model')
        axes[0, 1].set_ylabel('Percentage of Anomalies (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Time series with anomalies (using isolation forest results)
        if 'isolation_forest' in result['anomalies']:
            if_result = result['anomalies']['isolation_forest']
            if if_result['indices']:
                anomaly_dates = df.iloc[if_result['indices']]['Date']
                
                # Plot time series
                axes[1, 0].plot(df['Date'], df['emissions'], label='Emissions', alpha=0.7)
                axes[1, 0].scatter(anomaly_dates, df.iloc[if_result['indices']]['emissions'], 
                                 color='red', s=50, label='Anomalies', zorder=5)
                axes[1, 0].set_title('Time Series with Detected Anomalies')
                axes[1, 0].set_ylabel('Emissions')
                axes[1, 0].legend()
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Consensus anomalies (anomalies detected by multiple models)
        consensus_anomalies = self._find_consensus_anomalies(result['anomalies'])
        if consensus_anomalies:
            consensus_dates = df.iloc[consensus_anomalies]['Date']
            
            axes[1, 1].plot(df['Date'], df['emissions'], label='Emissions', alpha=0.7)
            axes[1, 1].scatter(consensus_dates, df.iloc[consensus_anomalies]['emissions'], 
                             color='darkred', s=70, label='Consensus Anomalies', zorder=5)
            axes[1, 1].set_title('Time Series with Consensus Anomalies')
            axes[1, 1].set_ylabel('Emissions')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = self.reports_dir / f"anomaly_plots_{entity}_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plots saved for {entity}: {plot_file}")
        
        plt.show()
    
    def _find_consensus_anomalies(self, anomalies: Dict) -> List[int]:
        """Find anomalies that are detected by multiple models."""
        all_anomaly_indices = []
        for model_result in anomalies.values():
            if 'indices' in model_result and model_result['indices']:
                all_anomaly_indices.extend(model_result['indices'])
        
        # Count occurrences
        from collections import Counter
        index_counts = Counter(all_anomaly_indices)
        
        # Return indices detected by at least 2 models
        consensus_indices = [idx for idx, count in index_counts.items() if count >= 2]
        return consensus_indices
    
    def analyze_anomaly_details(self, entity: str) -> Dict[str, Any]:
        """
        Provide detailed analysis of where anomalies are coming from.
        
        Args:
            entity: Entity name to analyze
            
        Returns:
            Dictionary containing detailed anomaly analysis
        """
        if entity not in self.results:
            logger.warning(f"No results available for {entity}")
            return {}
        
        # Load original data
        processed_dir = Path(self.config['data']['processed_dir'])
        df = pd.read_csv(processed_dir / f"cleaned_{entity}.csv")
        
        # Ensure Date column exists
        if 'Date' not in df.columns and 'fiscalyear' in df.columns:
            df['Date'] = pd.to_datetime(df['fiscalyear'].astype(str) + '-01-01')
        df['Date'] = pd.to_datetime(df['Date'])
        
        result = self.results[entity]
        analysis = {
            'entity': entity,
            'total_samples': len(df),
            'anomaly_details': {},
            'source_analysis': {},
            'property_analysis': {},
            'temporal_analysis': {},
            'consensus_analysis': {}
        }
        
        # Analyze each model's anomalies
        for model_name, model_result in result['anomalies'].items():
            if 'indices' in model_result and model_result['indices']:
                anomaly_indices = model_result['indices']
                anomaly_data = df.iloc[anomaly_indices]
                
                # Basic statistics
                analysis['anomaly_details'][model_name] = {
                    'count': len(anomaly_indices),
                    'percentage': len(anomaly_indices) / len(df) * 100,
                    'emissions_stats': {
                        'mean': float(anomaly_data['emissions'].mean()),
                        'std': float(anomaly_data['emissions'].std()),
                        'min': float(anomaly_data['emissions'].min()),
                        'max': float(anomaly_data['emissions'].max()),
                        'median': float(anomaly_data['emissions'].median())
                    }
                }
                
                # Source analysis (if available)
                if 'emissions_source' in df.columns:
                    source_counts = anomaly_data['emissions_source'].value_counts().to_dict()
                    analysis['source_analysis'][model_name] = source_counts
                
                # Property analysis (if available)
                property_cols = [col for col in df.columns if 'property' in col.lower() or 'building' in col.lower() or col in ['property_name', 'property_type', 'sector']]
                if property_cols:
                    property_analysis = {}
                    for prop_col in property_cols:
                        if prop_col in anomaly_data.columns:
                            prop_counts = anomaly_data[prop_col].value_counts().to_dict()
                            property_analysis[prop_col] = prop_counts
                    analysis['property_analysis'][model_name] = property_analysis
                
                # Temporal analysis
                temporal_analysis = {
                    'years': anomaly_data['Date'].dt.year.value_counts().to_dict(),
                    'months': anomaly_data['Date'].dt.month.value_counts().to_dict(),
                    'quarters': anomaly_data['Date'].dt.quarter.value_counts().to_dict()
                }
                analysis['temporal_analysis'][model_name] = temporal_analysis
        
        # Consensus analysis
        consensus_indices = self._find_consensus_anomalies(result['anomalies'])
        if consensus_indices:
            consensus_data = df.iloc[consensus_indices]
            
            analysis['consensus_analysis'] = {
                'count': len(consensus_indices),
                'percentage': len(consensus_indices) / len(df) * 100,
                'emissions_stats': {
                    'mean': float(consensus_data['emissions'].mean()),
                    'std': float(consensus_data['emissions'].std()),
                    'min': float(consensus_data['emissions'].min()),
                    'max': float(consensus_data['emissions'].max()),
                    'median': float(consensus_data['emissions'].median())
                }
            }
            
            # Source analysis for consensus anomalies
            if 'emissions_source' in df.columns:
                consensus_source_counts = consensus_data['emissions_source'].value_counts().to_dict()
                analysis['consensus_analysis']['sources'] = consensus_source_counts
            
            # Temporal analysis for consensus anomalies
            consensus_temporal = {
                'years': consensus_data['Date'].dt.year.value_counts().to_dict(),
                'months': consensus_data['Date'].dt.month.value_counts().to_dict(),
                'quarters': consensus_data['Date'].dt.quarter.value_counts().to_dict()
            }
            analysis['consensus_analysis']['temporal'] = consensus_temporal
        
        return analysis
    
    def generate_detailed_anomaly_report(self, entity: str) -> str:
        """
        Generate a detailed report showing exactly where anomalies are coming from.
        
        Args:
            entity: Entity name to analyze
            
        Returns:
            Path to the generated report file
        """
        analysis = self.analyze_anomaly_details(entity)
        
        if not analysis:
            logger.warning(f"No analysis available for {entity}")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"detailed_anomaly_analysis_{entity}_{timestamp}.md"
        
        report_content = f"""# Detailed Anomaly Analysis Report - {entity}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This report provides detailed analysis of where anomalies are coming from in the {entity} dataset.

## Summary Statistics
- **Total Samples**: {analysis['total_samples']}
- **Contamination Level**: 0.1% (very strict detection)

## Anomaly Detection Results by Model
"""
        
        # Add model-specific results
        for model_name, details in analysis['anomaly_details'].items():
            report_content += f"""
### {model_name.replace('_', ' ').title()}
- **Anomaly Count**: {details['count']} ({details['percentage']:.2f}%)
- **Emissions Statistics for Anomalies**:
  - Mean: {details['emissions_stats']['mean']:.2f}
  - Std: {details['emissions_stats']['std']:.2f}
  - Min: {details['emissions_stats']['min']:.2f}
  - Max: {details['emissions_stats']['max']:.2f}
  - Median: {details['emissions_stats']['median']:.2f}
"""
        
        # Add source analysis
        if analysis['source_analysis']:
            report_content += "\n## Emission Source Analysis\n"
            for model_name, source_counts in analysis['source_analysis'].items():
                report_content += f"\n### {model_name.replace('_', ' ').title()}\n"
                for source, count in source_counts.items():
                    report_content += f"- **{source}**: {count} anomalies\n"
        
        # Add property analysis
        if analysis['property_analysis']:
            report_content += "\n## Property Analysis\n"
            for model_name, property_data in analysis['property_analysis'].items():
                report_content += f"\n### {model_name.replace('_', ' ').title()}\n"
                for prop_col, prop_counts in property_data.items():
                    report_content += f"\n#### {prop_col}:\n"
                    for prop, count in prop_counts.items():
                        report_content += f"- **{prop}**: {count} anomalies\n"
        
        # Add temporal analysis
        if analysis['temporal_analysis']:
            report_content += "\n## Temporal Analysis\n"
            for model_name, temporal_data in analysis['temporal_analysis'].items():
                report_content += f"\n### {model_name.replace('_', ' ').title()}\n"
                
                # Years
                if temporal_data['years']:
                    report_content += "\n#### By Year:\n"
                    for year, count in sorted(temporal_data['years'].items()):
                        report_content += f"- **{year}**: {count} anomalies\n"
                
                # Months
                if temporal_data['months']:
                    report_content += "\n#### By Month:\n"
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    for month, count in sorted(temporal_data['months'].items()):
                        month_name = month_names[month - 1] if 1 <= month <= 12 else f"Month {month}"
                        report_content += f"- **{month_name}**: {count} anomalies\n"
                
                # Quarters
                if temporal_data['quarters']:
                    report_content += "\n#### By Quarter:\n"
                    for quarter, count in sorted(temporal_data['quarters'].items()):
                        report_content += f"- **Q{quarter}**: {count} anomalies\n"
        
        # Add consensus analysis
        if analysis['consensus_analysis']:
            consensus = analysis['consensus_analysis']
            report_content += f"""
## Consensus Anomalies (Detected by Multiple Models)
- **Count**: {consensus['count']} ({consensus['percentage']:.2f}%)
- **Emissions Statistics**:
  - Mean: {consensus['emissions_stats']['mean']:.2f}
  - Std: {consensus['emissions_stats']['std']:.2f}
  - Min: {consensus['emissions_stats']['min']:.2f}
  - Max: {consensus['emissions_stats']['max']:.2f}
  - Median: {consensus['emissions_stats']['median']:.2f}
"""
            
            # Consensus sources
            if 'sources' in consensus:
                report_content += "\n### Consensus Anomalies by Source:\n"
                for source, count in consensus['sources'].items():
                    report_content += f"- **{source}**: {count} anomalies\n"
            
            # Consensus temporal
            if 'temporal' in consensus:
                report_content += "\n### Consensus Anomalies by Time Period:\n"
                if consensus['temporal']['years']:
                    report_content += "\n#### By Year:\n"
                    for year, count in sorted(consensus['temporal']['years'].items()):
                        report_content += f"- **{year}**: {count} anomalies\n"
        
        # Add recommendations
        report_content += """
## Key Insights and Recommendations

### High-Priority Investigations:
1. **Consensus Anomalies**: Focus on anomalies detected by multiple models as they are most likely to be real issues
2. **Temporal Patterns**: Investigate time periods with high anomaly concentrations
3. **Source-Specific Issues**: Examine emission sources with disproportionate anomaly rates
4. **Property-Specific Issues**: Look into properties with unusual anomaly patterns

### Data Quality Actions:
1. **Verify Extreme Values**: Check if high-emission anomalies represent actual data or measurement errors
2. **Review Negative Emissions**: Investigate negative emission values for potential corrections
3. **Validate Source Data**: Ensure emission source classifications are accurate
4. **Cross-Reference Properties**: Verify property information consistency

### Operational Recommendations:
1. **Monitoring Focus**: Increase monitoring frequency for high-anomaly sources/properties
2. **Process Review**: Investigate operational processes during high-anomaly periods
3. **Equipment Checks**: Review equipment performance during anomaly periods
4. **Staff Training**: Provide additional training for data collection during critical periods
"""
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Detailed anomaly analysis report generated for {entity}: {report_file}")
        return str(report_file) 