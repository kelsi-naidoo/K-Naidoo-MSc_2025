"""
Test Setup Script for Carbon Emissions Forecasting System v3.0

This script tests the basic setup and data loading functionality.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import sys
from pathlib import Path
import json

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import openpyxl
        print("✓ OpenPyXL")
    except ImportError as e:
        print(f"✗ OpenPyXL import failed: {e}")
        return False
    
    return True

def test_config():
    """Test if configuration file can be loaded."""
    print("\nTesting configuration...")
    
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        print("✓ Configuration loaded successfully")
        print(f"  Project: {config['project']['name']} v{config['project']['version']}")
        print(f"  Entities: {config['data']['entities']}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_data_files():
    """Test if data files exist."""
    print("\nTesting data files...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    files_found = list(data_dir.glob("*.xlsx"))
    if not files_found:
        print(f"✗ No Excel files found in {data_dir}")
        return False
    
    print(f"✓ Found {len(files_found)} data files:")
    for file in files_found:
        print(f"  - {file.name}")
    
    return True

def test_basic_data_loading():
    """Test basic data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        # Try to load one file
        data_dir = Path("data/raw")
        files = list(data_dir.glob("*.xlsx"))
        
        if files:
            test_file = files[0]
            print(f"Testing with file: {test_file.name}")
            
            df = pd.read_excel(test_file, engine='openpyxl')
            print(f"✓ Successfully loaded data: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            # Check for key columns
            has_date = any('date' in col.lower() or 'month' in col.lower() for col in df.columns)
            has_emissions = any('emission' in col.lower() for col in df.columns)
            has_sector = any('sector' in col.lower() for col in df.columns)
            
            print(f"  Has date column: {has_date}")
            print(f"  Has emissions column: {has_emissions}")
            print(f"  Has sector column: {has_sector}")
            
            return True
        else:
            print("✗ No files to test")
            return False
            
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_directories():
    """Test if required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "reports/figures",
        "reports/logs",
        "config",
        "src",
        "notebooks"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print("=" * 60)
    print("CARBON AI V3.0 - SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Data Files", test_data_files),
        ("Data Loading", test_basic_data_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run the data exploration script:")
        print("   python notebooks/01_data_exploration.py")
        print("2. Run the data cleaning script:")
        print("   python notebooks/02_data_cleaning.py")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Run: python setup.py")
        print("2. Activate virtual environment")
        print("3. Install missing packages: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 