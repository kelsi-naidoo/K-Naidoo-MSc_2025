"""
Test Dashboard Functionality
Test script to verify the Streamlit dashboard components work correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append('src')

def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def test_entity_data_loading():
    """Test entity data loading."""
    print("\nTesting entity data loading...")
    
    config = test_config_loading()
    if not config:
        return None, None
    
    processed_dir = Path(config['data']['processed_dir'])
    
    # Test Entity A
    entity_a_file = processed_dir / 'cleaned_EntityA.csv'
    entity_a_data = None
    if entity_a_file.exists():
        try:
            entity_a_data = pd.read_csv(entity_a_file)
            if 'fiscalyear' in entity_a_data.columns:
                entity_a_data['fiscalyear'] = pd.to_datetime(entity_a_data['fiscalyear'])
            print(f"✅ Entity A data loaded: {entity_a_data.shape}")
        except Exception as e:
            print(f"❌ Error loading Entity A: {e}")
    else:
        print("⚠️ Entity A file not found")
    
    # Test Entity B
    entity_b_file = processed_dir / 'cleaned_EntityB.csv'
    entity_b_data = None
    if entity_b_file.exists():
        try:
            entity_b_data = pd.read_csv(entity_b_file)
            if 'fiscalyear' in entity_b_data.columns:
                entity_b_data['fiscalyear'] = pd.to_datetime(entity_b_data['fiscalyear'])
            print(f"✅ Entity B data loaded: {entity_b_data.shape}")
        except Exception as e:
            print(f"❌ Error loading Entity B: {e}")
    else:
        print("⚠️ Entity B file not found")
    
    return entity_a_data, entity_b_data

def test_performance_data():
    """Test performance data availability."""
    print("\nTesting performance data availability...")
    
    reports_dir = Path("reports")
    
    # Check model comparison files
    model_files = list(reports_dir.glob("*_model_comparison.csv"))
    if model_files:
        print(f"✅ Found {len(model_files)} model comparison files:")
        for file in model_files:
            print(f"   - {file.name}")
    else:
        print("⚠️ No model comparison files found")
    
    # Check performance metrics files
    perf_files = list(reports_dir.glob("*_performance_metrics.csv"))
    if perf_files:
        print(f"✅ Found {len(perf_files)} performance metrics files:")
        for file in perf_files:
            print(f"   - {file.name}")
    else:
        print("⚠️ No performance metrics files found")
    
    return len(model_files) > 0, len(perf_files) > 0

def test_dashboard_imports():
    """Test that all dashboard imports work."""
    print("\nTesting dashboard imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        print("✅ Plotly Express imported successfully")
    except ImportError as e:
        print(f"❌ Plotly Express import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ Plotly Graph Objects imported successfully")
    except ImportError as e:
        print(f"❌ Plotly Graph Objects import failed: {e}")
        return False
    
    return True

def test_app_module():
    """Test that the app module can be imported."""
    print("\nTesting app module import...")
    
    try:
        # Test importing the app functions
        import app
        print("✅ App module imported successfully")
        
        # Test key functions
        if hasattr(app, 'load_config'):
            print("✅ load_config function found")
        if hasattr(app, 'load_entity_data'):
            print("✅ load_entity_data function found")
        if hasattr(app, 'validate_uploaded_file'):
            print("✅ validate_uploaded_file function found")
        if hasattr(app, 'upload_file'):
            print("✅ upload_file function found")
        if hasattr(app, 'display_preloaded_data'):
            print("✅ display_preloaded_data function found")
        if hasattr(app, 'main'):
            print("✅ main function found")
        
        return True
    except Exception as e:
        print(f"❌ App module import failed: {e}")
        return False

def main():
    """Run all dashboard tests."""
    print("=" * 60)
    print("DASHBOARD FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_dashboard_imports()
    
    # Test app module
    app_ok = test_app_module()
    
    # Test configuration
    config_ok = test_config_loading() is not None
    
    # Test data loading
    entity_a, entity_b = test_entity_data_loading()
    data_ok = entity_a is not None or entity_b is not None
    
    # Test performance data
    model_data, perf_data = test_performance_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"✅ App Module: {'PASS' if app_ok else 'FAIL'}")
    print(f"✅ Configuration: {'PASS' if config_ok else 'FAIL'}")
    print(f"✅ Data Loading: {'PASS' if data_ok else 'FAIL'}")
    print(f"✅ Model Data: {'PASS' if model_data else 'WARNING'}")
    print(f"✅ Performance Data: {'PASS' if perf_data else 'WARNING'}")
    
    if imports_ok and app_ok and config_ok:
        print("\n🎉 Dashboard is ready to run!")
        print("Run 'python run_dashboard.py' to launch the dashboard")
    else:
        print("\n❌ Dashboard has issues that need to be resolved")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 