"""
03 - Forecasting Models Script
Carbon Emissions Forecasting System v3.0

This script implements multiple forecasting models for carbon emissions prediction:
- ARIMA (AutoRegressive Integrated Moving Average)
- LSTM (Long Short-Term Memory)
- Linear Regression
- XGBoost

Author: Kelsi Naidoo
Institution: University of Cape Town
Date: June 2025

IEEE Standards Compliance:
- IEEE 1012-2016: Model validation and verification
- IEEE 730-2014: Quality assurance procedures
- IEEE 829-2008: Test documentation

Objectives:
1. Load and prepare cleaned data for modeling
2. Implement multiple forecasting models
3. Train and validate models with proper evaluation metrics
4. Generate forecasts and confidence intervals
5. Compare model performance and select best model
6. Save results for dashboard integration
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
from typing import Dict, List, Tuple, Optional
import time
import psutil
import os
import gc

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM models will be skipped.")
    TENSORFLOW_AVAILABLE = False

# Add src to path for importing our modules
sys.path.append('src')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)

print(f"Script executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")

# Load configuration
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Set up paths
PROCESSED_DATA_DIR = Path(config['data']['processed_dir'])
FIGURES_DIR = Path(config['reports']['figures_dir'])
REPORTS_DIR = Path(config['reports']['output_dir'])

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"Figures directory: {FIGURES_DIR}")
print(f"Reports directory: {REPORTS_DIR}")

class PerformanceMonitor:
    """Monitor and track computational performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process(os.getpid())
        
    def start_monitoring(self, model_name: str, entity_name: str):
        """Start monitoring for a specific model."""
        key = f"{entity_name}_{model_name}"
        self.metrics[key] = {
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
            'start_cpu_percent': self.process.cpu_percent(),
            'peak_memory': 0,
            'peak_cpu_percent': 0,
            'total_memory_used': 0,
            'memory_samples': [],
            'cpu_samples': []
        }
        print(f"Started monitoring {model_name} for {entity_name}")
        
    def update_monitoring(self, model_name: str, entity_name: str):
        """Update monitoring metrics during training."""
        key = f"{entity_name}_{model_name}"
        if key in self.metrics:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            current_cpu = self.process.cpu_percent()
            
            self.metrics[key]['memory_samples'].append(current_memory)
            self.metrics[key]['cpu_samples'].append(current_cpu)
            
            self.metrics[key]['peak_memory'] = max(self.metrics[key]['peak_memory'], current_memory)
            self.metrics[key]['peak_cpu_percent'] = max(self.metrics[key]['peak_cpu_percent'], current_cpu)
    
    def stop_monitoring(self, model_name: str, entity_name: str) -> Dict:
        """Stop monitoring and return performance metrics."""
        key = f"{entity_name}_{model_name}"
        if key in self.metrics:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            metrics = self.metrics[key]
            metrics['end_time'] = end_time
            metrics['end_memory'] = end_memory
            metrics['execution_time'] = end_time - metrics['start_time']
            metrics['memory_increase'] = end_memory - metrics['start_memory']
            metrics['total_memory_used'] = end_memory - metrics['start_memory']
            
            # Calculate average memory and CPU usage
            if metrics['memory_samples']:
                metrics['avg_memory'] = np.mean(metrics['memory_samples'])
                metrics['avg_cpu_percent'] = np.mean(metrics['cpu_samples'])
            
            # Estimate computational cost (rough approximation)
            # CPU seconds = execution_time * avg_cpu_percent / 100
            metrics['cpu_seconds'] = metrics['execution_time'] * metrics.get('avg_cpu_percent', 0) / 100
            
            # Memory cost (MB-seconds)
            metrics['memory_seconds'] = metrics['avg_memory'] * metrics['execution_time']
            
            print(f"Performance for {model_name} ({entity_name}):")
            print(f"  Execution time: {metrics['execution_time']:.2f}s")
            print(f"  Peak memory: {metrics['peak_memory']:.1f}MB")
            print(f"  Peak CPU: {metrics['peak_cpu_percent']:.1f}%")
            print(f"  Memory increase: {metrics['memory_increase']:.1f}MB")
            
            return metrics
        return {}

def clean_features(X, y):
    """Simple cleaning: replace inf/-inf with NaN, drop rows with NaN."""
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean

class EmissionsForecaster:
    """Comprehensive emissions forecasting system."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.results = {}
        self.feature_importance = {}
        self.performance_monitor = PerformanceMonitor()
        self.performance_metrics = {}
        
    def load_and_prepare_data(self, entity_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for forecasting."""
        print(f"\nLoading data for {entity_name}...")
        
        # Load feature-engineered data
        features_file = PROCESSED_DATA_DIR / f'features_{entity_name}.csv'
        if features_file.exists():
            df = pd.read_csv(features_file)
            print(f"Loaded features data: {df.shape}")
        else:
            # Fallback to cleaned data
            cleaned_file = PROCESSED_DATA_DIR / f'cleaned_{entity_name}.csv'
            df = pd.read_csv(cleaned_file)
            print(f"Loaded cleaned data: {df.shape}")
        
        # Convert fiscalyear to datetime
        if 'fiscalyear' in df.columns:
            df['fiscalyear'] = pd.to_datetime(df['fiscalyear'])
        
        # Sort by date
        df = df.sort_values('fiscalyear').reset_index(drop=True)
        
        # Prepare features and target
        feature_columns = [col for col in df.columns 
                          if col not in ['fiscalyear', 'emissions', 'emissions_original', 
                                       'is_correction', 'correction_amount']]
        
        # Handle categorical features
        categorical_features = []
        for col in feature_columns:
            if df[col].dtype == 'object':
                categorical_features.append(col)
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[f'{col}_{entity_name}'] = le
        
        # Select final feature columns (numeric only)
        final_features = [col for col in df.columns 
                         if col.endswith('_encoded') or 
                         (df[col].dtype in ['int64', 'float64'] and col in feature_columns)]
        
        X = df[final_features].fillna(0)
        y = df['emissions'].fillna(0)
        
        print(f"Features: {len(final_features)}")
        print(f"Target shape: {y.shape}")
        print(f"Date range: {df['fiscalyear'].min()} to {df['fiscalyear'].max()}")
        
        # Clean features for ML models
        X, y = clean_features(X, y)
        
        return X, y, df
    
    def prepare_time_series_data(self, df: pd.DataFrame, target_col: str = 'emissions') -> pd.Series:
        """Prepare time series data for ARIMA models."""
        # Aggregate by date (sum emissions per day)
        ts_data = df.groupby('fiscalyear')[target_col].sum().sort_index()
        
        # Resample to monthly frequency if needed
        if len(ts_data) > 100:  # If too many data points, resample to monthly
            ts_data = ts_data.resample('M').sum()
        
        print(f"Time series data shape: {ts_data.shape}")
        print(f"Time series range: {ts_data.index.min()} to {ts_data.index.max()}")
        
        return ts_data
    
    def check_stationarity(self, ts_data: pd.Series) -> Dict:
        """Check if time series is stationary."""
        result = adfuller(ts_data.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Is stationary: {stationarity_result['is_stationary']}")
        
        return stationarity_result
    
    def train_arima_model(self, ts_data: pd.Series, entity_name: str) -> Dict:
        """Train ARIMA model."""
        print(f"\nTraining ARIMA model for {entity_name}...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring('arima', entity_name)
        
        # Check stationarity
        stationarity = self.check_stationarity(ts_data)
        
        # Make series stationary if needed
        if not stationarity['is_stationary']:
            ts_data_diff = ts_data.diff().dropna()
            print("Applied differencing to make series stationary")
        else:
            ts_data_diff = ts_data
        
        # Handle small datasets - ensure minimum test size
        min_test_size = 2
        if len(ts_data_diff) < 10:
            # For very small datasets, use more data for training
            train_size = max(3, len(ts_data_diff) - min_test_size)
        else:
            train_size = int(len(ts_data_diff) * 0.8)
        
        train = ts_data_diff[:train_size]
        test = ts_data_diff[train_size:]
        
        print(f"Training on {len(train)} points, testing on {len(test)} points")
        
        # Try different ARIMA parameters
        best_aic = float('inf')
        best_params = None
        best_model = None
        
        # Grid search for best parameters (simplified for small datasets)
        if len(train) < 10:
            p_values = range(0, 2)
            d_values = range(0, 2)
            q_values = range(0, 2)
        else:
            p_values = range(0, 3)
            d_values = range(0, 2)
            q_values = range(0, 3)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(train, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        # Update monitoring during grid search
                        self.performance_monitor.update_monitoring('arima', entity_name)
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                            best_model = fitted_model
                            
                    except:
                        continue
        
        if best_model is None:
            print("Could not fit ARIMA model, using default parameters")
            best_model = ARIMA(train, order=(1, 1, 1)).fit()
            best_params = (1, 1, 1)
        
        # Make predictions
        if len(test) > 0:
            forecast = best_model.forecast(steps=len(test))
        else:
            # If no test data, forecast next few periods
            forecast = best_model.forecast(steps=3)
            test = pd.Series([np.nan] * 3, index=pd.date_range(ts_data_diff.index[-1], periods=4, freq='Y')[1:])
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mse = mean_squared_error(test, forecast)
        
        # Calculate R² for ARIMA
        ss_res = np.sum((test - forecast) ** 2)
        ss_tot = np.sum((test - np.mean(test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        print(f"ARIMA({best_params[0]},{best_params[1]},{best_params[2]}) - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Stop performance monitoring and get metrics
        performance_metrics = self.performance_monitor.stop_monitoring('arima', entity_name)
        self.performance_metrics[f'arima_{entity_name}'] = performance_metrics
        
        return {
            'train': train,
            'test': test,
            'forecast': forecast,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'stationarity': stationarity,
            'performance_metrics': performance_metrics
        }
    
    def train_lstm_model(self, X: pd.DataFrame, y: pd.Series, entity_name: str) -> Dict:
        """Train LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available, skipping LSTM model")
            return None
        
        print(f"\nTraining LSTM model for {entity_name}...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring('lstm', entity_name)
        
        # For large datasets, sample a subset to speed up training
        if len(X) > 10000:
            print(f"Large dataset detected ({len(X)} samples). Sampling 10,000 samples for faster training...")
            sample_indices = np.random.choice(len(X), 10000, replace=False)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        # Prepare data for LSTM (3D format: [samples, timesteps, features])
        def create_sequences(X, y, time_steps=12):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X.iloc[i:(i + time_steps)].values)
                ys.append(y.iloc[i + time_steps])
            return np.array(Xs), np.array(ys)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        self.scalers[f'lstm_{entity_name}'] = scaler
        
        # Create sequences with smaller time steps for faster training
        time_steps = min(6, len(X_scaled) // 20)  # Reduced time steps
        X_seq, y_seq = create_sequences(pd.DataFrame(X_scaled), y_sample, time_steps)
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Note: X_train and X_test are numpy arrays, not pandas DataFrames
        # The data is already clean from the sequence creation process
        
        # Build simplified LSTM model for faster training
        model = Sequential([
            LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            Dense(16),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Custom callback to monitor performance during training
        class PerformanceCallback(tf.keras.callbacks.Callback):
            def __init__(self, monitor, entity_name):
                super().__init__()
                self.monitor = monitor
                self.entity_name = entity_name
            
            def on_epoch_end(self, epoch, logs=None):
                self.monitor.update_monitoring('lstm', self.entity_name)
        
        # Early stopping with shorter patience
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        performance_callback = PerformanceCallback(self.performance_monitor, entity_name)
        
        # Train model with fewer epochs
        history = model.fit(
            X_train, y_train,
            epochs=50,  # Reduced from 100
            batch_size=64,  # Increased batch size for faster training
            validation_split=0.2,
            callbacks=[early_stopping, performance_callback],
            verbose=1  # Show progress
        )
        
        # Make predictions
        y_pred = model.predict(X_test).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Stop performance monitoring and get metrics
        performance_metrics = self.performance_monitor.stop_monitoring('lstm', entity_name)
        self.performance_metrics[f'lstm_{entity_name}'] = performance_metrics
        
        lstm_results = {
            'model': model,
            'scaler': scaler,
            'history': history,
            'predictions': y_pred,
            'actual': y_test,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'performance_metrics': performance_metrics
        }
        
        print(f"LSTM - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Clean up memory
        del model
        gc.collect()
        
        return lstm_results
    
    def train_regression_model(self, X: pd.DataFrame, y: pd.Series, entity_name: str) -> Dict:
        """Train Linear Regression model."""
        print(f"\nTraining Linear Regression model for {entity_name}...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring('regression', entity_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[f'regression_{entity_name}'] = scaler
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Update monitoring during training
        self.performance_monitor.update_monitoring('regression', entity_name)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        # Stop performance monitoring and get metrics
        performance_metrics = self.performance_monitor.stop_monitoring('regression', entity_name)
        self.performance_metrics[f'regression_{entity_name}'] = performance_metrics
        
        regression_results = {
            'model': model,
            'scaler': scaler,
            'predictions': y_pred,
            'actual': y_test,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'performance_metrics': performance_metrics
        }
        
        print(f"Linear Regression - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return regression_results
    
    def train_xgboost_model(self, X: pd.DataFrame, y: pd.Series, entity_name: str) -> Dict:
        """Train XGBoost model."""
        print(f"\nTraining XGBoost model for {entity_name}...")
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring('xgboost', entity_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Update monitoring during training
        self.performance_monitor.update_monitoring('xgboost', entity_name)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Stop performance monitoring and get metrics
        performance_metrics = self.performance_monitor.stop_monitoring('xgboost', entity_name)
        self.performance_metrics[f'xgboost_{entity_name}'] = performance_metrics
        
        xgboost_results = {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'feature_importance': feature_importance,
            'performance_metrics': performance_metrics
        }
        
        print(f"XGBoost - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return xgboost_results
    
    def compare_models(self, entity_name: str) -> pd.DataFrame:
        """Compare performance of all models and report the best one based on RMSE."""
        print(f"\nComparing models for {entity_name}...")
        
        model_results = self.results.get(entity_name, {})
        model_comparison = []
        
        # Add ARIMA results
        if 'arima' in model_results and model_results['arima'] is not None:
            arima_results = model_results['arima']
            model_comparison.append({
                'Model': 'ARIMA',
                'RMSE': arima_results['rmse'],
                'MAE': arima_results['mae'],
                'MSE': arima_results['mse'],
                'R²': arima_results['r2'],
                'Execution_Time': arima_results.get('performance_metrics', {}).get('execution_time', np.nan),
                'Peak_Memory_MB': arima_results.get('performance_metrics', {}).get('peak_memory', np.nan),
                'Peak_CPU_Percent': arima_results.get('performance_metrics', {}).get('peak_cpu_percent', np.nan)
            })
        
        # Add LSTM results
        if 'lstm' in model_results and model_results['lstm'] is not None:
            lstm_results = model_results['lstm']
            model_comparison.append({
                'Model': 'LSTM',
                'RMSE': lstm_results.get('rmse', np.nan),
                'MAE': lstm_results.get('mae', np.nan),
                'MSE': lstm_results.get('mse', np.nan),
                'R²': lstm_results.get('r2', np.nan),
                'Execution_Time': lstm_results.get('performance_metrics', {}).get('execution_time', np.nan),
                'Peak_Memory_MB': lstm_results.get('performance_metrics', {}).get('peak_memory', np.nan),
                'Peak_CPU_Percent': lstm_results.get('performance_metrics', {}).get('peak_cpu_percent', np.nan)
            })
        
        # Add Regression results
        if 'regression' in model_results and model_results['regression'] is not None:
            reg_results = model_results['regression']
            model_comparison.append({
                'Model': 'REGRESSION',
                'RMSE': reg_results.get('rmse', np.nan),
                'MAE': reg_results.get('mae', np.nan),
                'MSE': reg_results.get('mse', np.nan),
                'R²': reg_results.get('r2', np.nan),
                'Execution_Time': reg_results.get('performance_metrics', {}).get('execution_time', np.nan),
                'Peak_Memory_MB': reg_results.get('performance_metrics', {}).get('peak_memory', np.nan),
                'Peak_CPU_Percent': reg_results.get('performance_metrics', {}).get('peak_cpu_percent', np.nan)
            })
        
        # Add XGBoost results
        if 'xgboost' in model_results and model_results['xgboost'] is not None:
            xgb_results = model_results['xgboost']
            model_comparison.append({
                'Model': 'XGBOOST',
                'RMSE': xgb_results.get('rmse', np.nan),
                'MAE': xgb_results.get('mae', np.nan),
                'MSE': xgb_results.get('mse', np.nan),
                'R²': xgb_results.get('r2', np.nan),
                'Execution_Time': xgb_results.get('performance_metrics', {}).get('execution_time', np.nan),
                'Peak_Memory_MB': xgb_results.get('performance_metrics', {}).get('peak_memory', np.nan),
                'Peak_CPU_Percent': xgb_results.get('performance_metrics', {}).get('peak_cpu_percent', np.nan)
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(model_comparison)
        print(f"\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Select and report the best model based on RMSE
        if not comparison_df.empty:
            best_model_row = comparison_df.loc[comparison_df['RMSE'].idxmin()]
            print(f"\nBest model for {entity_name}: {best_model_row['Model']} (RMSE: {best_model_row['RMSE']:.4f}, R²: {best_model_row['R²']})")
            # Save best model info for summary
            self.results[entity_name]['best_model'] = best_model_row.to_dict()
        else:
            print(f"\nNo valid models found for {entity_name}.")
        
        return comparison_df
    
    def plot_performance_comparison(self, entity_name: str):
        """Create performance comparison plots."""
        print(f"\nGenerating performance comparison plots for {entity_name}...")
        
        comparison_df = self.compare_models(entity_name)
        if comparison_df.empty:
            print("No performance data available for plotting")
            return
        
        # Create performance comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Performance Comparison - {entity_name}', fontsize=16)
        
        # Plot 1: Execution Time
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Execution_Time'])
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Peak Memory Usage
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Peak_Memory_MB'])
        axes[0, 1].set_title('Peak Memory Usage')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Peak CPU Usage
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Peak_CPU_Percent'])
        axes[1, 0].set_title('Peak CPU Usage')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: RMSE vs Execution Time (efficiency plot)
        scatter = axes[1, 1].scatter(comparison_df['Execution_Time'], comparison_df['RMSE'], 
                                   s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Execution Time (seconds)')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Efficiency: RMSE vs Execution Time')
        
        # Add model labels to scatter plot
        for i, model in enumerate(comparison_df['Model']):
            axes[1, 1].annotate(model, 
                              (comparison_df['Execution_Time'].iloc[i], comparison_df['RMSE'].iloc[i]),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = FIGURES_DIR / f'{entity_name}_performance_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance comparison figure: {fig_path}")
        plt.close(fig)
    
    def plot_results(self, entity_name: str):
        """Plot model results and comparisons."""
        print(f"\nGenerating plots for {entity_name}...")
        
        # Check which models succeeded
        model_results = self.results.get(entity_name, {})
        available_models = [k for k, v in model_results.items() if v is not None and 'rmse' in v]
        n_plots = 1 + int('arima' in model_results) + int('regression' in model_results) + int('xgboost' in model_results)
        nrows = 2 if n_plots > 2 else 1
        ncols = 2 if n_plots > 1 else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
        if n_plots == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        plot_idx = 0
        fig.suptitle(f'Forecasting Results - {entity_name}', fontsize=16)
        
        # Plot 1: Model comparison
        if entity_name in self.results:
            comparison_df = self.compare_models(entity_name)
            axes[plot_idx].bar(comparison_df['Model'], comparison_df['RMSE'])
            axes[plot_idx].set_title('Model RMSE Comparison')
            axes[plot_idx].set_ylabel('RMSE')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1
        # Plot 2: ARIMA results
        if 'arima' in model_results and model_results['arima'] is not None:
            arima_results = model_results['arima']
            
            # Plot training data
            if len(arima_results['train']) > 0:
                axes[plot_idx].plot(arima_results['train'].index, arima_results['train'].values, 
                                   label='Training', color='blue', marker='o', linewidth=2, markersize=6)
            
            # Plot test data and forecast
            if len(arima_results['test']) > 0:
                # For small datasets, show both actual and forecast
                if len(arima_results['test']) == 1:
                    # Single point - show as scatter with clear markers
                    axes[plot_idx].scatter(arima_results['test'].index, arima_results['test'].values, 
                                         label='Actual (Test)', color='red', s=100, zorder=5)
                    axes[plot_idx].scatter(arima_results['test'].index, arima_results['forecast'], 
                                         label='Forecast', color='green', s=100, marker='s', zorder=5)
                else:
                    # Multiple points - show as lines
                    axes[plot_idx].plot(arima_results['test'].index, arima_results['test'].values, 
                                       label='Actual (Test)', color='red', marker='o', linewidth=2, markersize=8)
                    axes[plot_idx].plot(arima_results['test'].index, arima_results['forecast'], 
                                       label='Forecast', color='green', marker='s', linewidth=2, markersize=8)
            
            axes[plot_idx].set_title(f'ARIMA Forecast vs Actual\n({len(arima_results["train"])} train, {len(arima_results["test"])} test)')
            axes[plot_idx].set_xlabel('Date')
            axes[plot_idx].set_ylabel('Emissions')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        # Plot 3: Regression results
        if 'regression' in model_results and model_results['regression'] is not None:
            reg_results = model_results['regression']
            axes[plot_idx].scatter(reg_results['actual'], reg_results['predictions'], alpha=0.5)
            axes[plot_idx].plot([reg_results['actual'].min(), reg_results['actual'].max()], 
                               [reg_results['actual'].min(), reg_results['actual'].max()], 'r--')
            axes[plot_idx].set_xlabel('Actual')
            axes[plot_idx].set_ylabel('Predicted')
            axes[plot_idx].set_title('Regression: Predicted vs Actual')
            plot_idx += 1
        # Plot 4: Feature importance (XGBoost)
        if 'xgboost' in model_results and model_results['xgboost'] is not None:
            xgb_results = model_results['xgboost']
            top_features = xgb_results['feature_importance'].head(10)
            axes[plot_idx].barh(range(len(top_features)), top_features['importance'])
            axes[plot_idx].set_yticks(range(len(top_features)))
            axes[plot_idx].set_yticklabels(top_features['feature'])
            axes[plot_idx].set_title('XGBoost Feature Importance (Top 10)')
            plot_idx += 1
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = FIGURES_DIR / f'{entity_name}_forecasting_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
        plt.close(fig)
        
        # Generate performance comparison plots
        self.plot_performance_comparison(entity_name)
    
    def save_results(self, entity_name: str):
        """Save model results and predictions."""
        print(f"\nSaving results for {entity_name}...")
        
        # Save model comparison
        comparison_df = self.compare_models(entity_name)
        comparison_df.to_csv(REPORTS_DIR / f'{entity_name}_model_comparison.csv', index=False)
        
        # Save performance metrics
        performance_summary = []
        for model_name, results in self.results[entity_name].items():
            if results is not None and 'performance_metrics' in results:
                perf_metrics = results['performance_metrics']
                performance_summary.append({
                    'Model': model_name.upper(),
                    'Entity': entity_name,
                    'Execution_Time_Seconds': perf_metrics.get('execution_time', np.nan),
                    'Peak_Memory_MB': perf_metrics.get('peak_memory', np.nan),
                    'Peak_CPU_Percent': perf_metrics.get('peak_cpu_percent', np.nan),
                    'Avg_Memory_MB': perf_metrics.get('avg_memory', np.nan),
                    'Avg_CPU_Percent': perf_metrics.get('avg_cpu_percent', np.nan),
                    'Memory_Increase_MB': perf_metrics.get('memory_increase', np.nan),
                    'CPU_Seconds': perf_metrics.get('cpu_seconds', np.nan),
                    'Memory_Seconds': perf_metrics.get('memory_seconds', np.nan)
                })
        
        if performance_summary:
            perf_df = pd.DataFrame(performance_summary)
            perf_df.to_csv(REPORTS_DIR / f'{entity_name}_performance_metrics.csv', index=False)
            print(f"Saved performance metrics: {REPORTS_DIR / f'{entity_name}_performance_metrics.csv'}")
        
        # Save predictions for each model
        for model_name, results in self.results[entity_name].items():
            if results is not None and 'predictions' in results:
                pred_df = pd.DataFrame({
                    'actual': results['actual'],
                    'predicted': results['predictions']
                })
                pred_df.to_csv(REPORTS_DIR / f'{entity_name}_{model_name}_predictions.csv', index=False)
        
        # Save feature importance
        for model_name, results in self.results[entity_name].items():
            if results is not None and 'feature_importance' in results:
                results['feature_importance'].to_csv(
                    REPORTS_DIR / f'{entity_name}_{model_name}_feature_importance.csv', 
                    index=False
                )
        
        print(f"Results saved to {REPORTS_DIR}")

    def print_final_summary(self):
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        # Performance summary
        print("\nPERFORMANCE SUMMARY:")
        print("-" * 40)
        all_performance = []
        for entity_name in self.results:
            for model_name, results in self.results[entity_name].items():
                if results is not None and 'performance_metrics' in results:
                    perf = results['performance_metrics']
                    all_performance.append({
                        'Entity': entity_name,
                        'Model': model_name.upper(),
                        'Execution_Time': perf.get('execution_time', np.nan),
                        'Peak_Memory_MB': perf.get('peak_memory', np.nan),
                        'Peak_CPU_Percent': perf.get('peak_cpu_percent', np.nan),
                        'RMSE': results.get('rmse', np.nan)
                    })
        
        if all_performance:
            perf_df = pd.DataFrame(all_performance)
            print("\nPerformance Metrics:")
            print(perf_df.to_string(index=False))
            
            # Find fastest and most memory efficient models
            fastest = perf_df.loc[perf_df['Execution_Time'].idxmin()]
            most_memory_efficient = perf_df.loc[perf_df['Peak_Memory_MB'].idxmin()]
            most_accurate = perf_df.loc[perf_df['RMSE'].idxmin()]
            
            print(f"\nFastest Model: {fastest['Model']} ({fastest['Entity']}) - {fastest['Execution_Time']:.2f}s")
            print(f"Most Memory Efficient: {most_memory_efficient['Model']} ({most_memory_efficient['Entity']}) - {most_memory_efficient['Peak_Memory_MB']:.1f}MB")
            print(f"Most Accurate: {most_accurate['Model']} ({most_accurate['Entity']}) - RMSE: {most_accurate['RMSE']:.4f}")
        
        # Model accuracy summary
        print("\nMODEL ACCURACY SUMMARY:")
        print("-" * 40)
        for entity_name in self.results:
            best = self.results[entity_name].get('best_model', None)
            if best:
                print(f"Best model for {entity_name}: {best['Model']} (RMSE: {best['RMSE']:.4f}, R²: {best['R²']})")
            else:
                print(f"No valid best model for {entity_name}.")
        
        # Computational cost summary
        print("\nCOMPUTATIONAL COST SUMMARY:")
        print("-" * 40)
        if all_performance:
            total_time = perf_df['Execution_Time'].sum()
            total_memory = perf_df['Peak_Memory_MB'].max()  # Peak across all models
            avg_cpu = perf_df['Peak_CPU_Percent'].mean()
            
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Peak memory usage: {total_memory:.1f} MB")
            print(f"Average peak CPU usage: {avg_cpu:.1f}%")
            
            # Cost efficiency analysis
            print(f"\nCost Efficiency Analysis:")
            perf_df['Efficiency_Score'] = perf_df['RMSE'] * perf_df['Execution_Time']  # Lower is better
            most_efficient = perf_df.loc[perf_df['Efficiency_Score'].idxmin()]
            print(f"Most cost-efficient model: {most_efficient['Model']} ({most_efficient['Entity']})")
            print(f"  - RMSE: {most_efficient['RMSE']:.4f}")
            print(f"  - Time: {most_efficient['Execution_Time']:.2f}s")
            print(f"  - Efficiency Score: {most_efficient['Efficiency_Score']:.4f}")

def main():
    """Main execution function."""
    print("="*80)
    print("CARBON EMISSIONS FORECASTING MODELS")
    print("="*80)
    
    # Initialize forecaster
    forecaster = EmissionsForecaster()
    
    # Process each entity
    for entity in config['data']['entities']:
        print(f"\n{'='*60}")
        print(f"PROCESSING ENTITY: {entity}")
        print(f"{'='*60}")
        
        try:
            # Load and prepare data
            X, y, df = forecaster.load_and_prepare_data(entity)
            
            # Initialize results dictionary for this entity
            forecaster.results[entity] = {}
            
            # Train ARIMA model
            try:
                ts_data = forecaster.prepare_time_series_data(df)
                arima_results = forecaster.train_arima_model(ts_data, entity)
                forecaster.results[entity]['arima'] = arima_results
            except Exception as e:
                print(f"ARIMA training failed: {e}")
                forecaster.results[entity]['arima'] = None
            
            # Train LSTM model
            try:
                lstm_results = forecaster.train_lstm_model(X, y, entity)
                forecaster.results[entity]['lstm'] = lstm_results
            except Exception as e:
                print(f"LSTM training failed: {e}")
                forecaster.results[entity]['lstm'] = None
            
            # Train Regression model
            try:
                regression_results = forecaster.train_regression_model(X, y, entity)
                forecaster.results[entity]['regression'] = regression_results
            except Exception as e:
                print(f"Regression training failed: {e}")
                forecaster.results[entity]['regression'] = None
            
            # Train XGBoost model
            try:
                xgboost_results = forecaster.train_xgboost_model(X, y, entity)
                forecaster.results[entity]['xgboost'] = xgboost_results
            except Exception as e:
                print(f"XGBoost training failed: {e}")
                forecaster.results[entity]['xgboost'] = None
            
            # Compare models
            comparison_df = forecaster.compare_models(entity)
            
            # Generate plots
            forecaster.plot_results(entity)
            
            # Save results
            forecaster.save_results(entity)
            
        except Exception as e:
            print(f"Error processing {entity}: {e}")
            continue
    
    # Generate final summary
    forecaster.print_final_summary()
    
    print(f"\nAll results saved to: {REPORTS_DIR}")
    print(f"All figures saved to: {FIGURES_DIR}")
    print("\nForecasting models completed successfully!")

if __name__ == "__main__":
    main() 