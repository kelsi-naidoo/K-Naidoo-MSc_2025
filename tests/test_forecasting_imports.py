#!/usr/bin/env python3
"""
Test script to replicate the exact import structure from forecasting script
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
    print("✓ TensorFlow imports successful")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

# Add src to path for importing our modules
sys.path.append('src')

print(f"Script executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")

# Test if we can create a simple LSTM model
if TENSORFLOW_AVAILABLE:
    try:
        # Create a simple LSTM model
        model = Sequential([
            LSTM(10, input_shape=(10, 5)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✓ LSTM model creation successful")
    except Exception as e:
        print(f"✗ LSTM model creation failed: {e}")
        TENSORFLOW_AVAILABLE = False

print(f"Final TENSORFLOW_AVAILABLE: {TENSORFLOW_AVAILABLE}") 