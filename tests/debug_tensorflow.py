#!/usr/bin/env python3
"""
Debug script to test TensorFlow imports
"""

import sys
print(f"Python path: {sys.path}")
print(f"Python executable: {sys.executable}")

# Test TensorFlow import
try:
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully: {tf.__version__}")
    
    # Test Keras imports
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        print("✓ All Keras imports successful")
        TENSORFLOW_AVAILABLE = True
    except ImportError as e:
        print(f"✗ Keras import failed: {e}")
        TENSORFLOW_AVAILABLE = False
        
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False

print(f"Final TENSORFLOW_AVAILABLE: {TENSORFLOW_AVAILABLE}") 