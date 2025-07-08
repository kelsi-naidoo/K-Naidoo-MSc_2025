"""
Test script for anomaly detection functionality.

This script tests the AnomalyDetector module to ensure it works correctly
with the existing data pipeline.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import sys
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_anomaly_detector():
    """Test the anomaly detection functionality."""
    print("=" * 60)
    print("TESTING ANOMALY DETECTION MODULE")
    print("=" * 60)
    
    try:
        # Test 1: Import AnomalyDetector
        print("\n1. Testing import...")
        from src.models.anomaly_detector import AnomalyDetector
        print("✓ AnomalyDetector imported successfully")
        
        # Test 2: Initialize detector
        print("\n2. Testing initialization...")
        config_path = project_root / "config" / "config.json"
        detector = AnomalyDetector(str(config_path))
        print("✓ AnomalyDetector initialized successfully")
        
        # Test 3: Load cleaned data
        print("\n3. Testing data loading...")
        cleaned_data = detector.load_cleaned_data()
        
        if cleaned_data:
            for entity, df in cleaned_data.items():
                if df is not None:
                    print(f"✓ Loaded {entity}: {df.shape[0]} rows, {df.shape[1]} columns")
                else:
                    print(f"✗ No data for {entity}")
        else:
            print("✗ No cleaned data found")
            return False
        
        # Test 4: Test anomaly detection on first available entity
        print("\n4. Testing anomaly detection...")
        for entity, df in cleaned_data.items():
            if df is not None and len(df) > 0:
                print(f"Testing anomaly detection on {entity}...")
                result = detector.detect_anomalies(df, entity)
                
                if result and 'anomalies' in result:
                    print(f"✓ Anomaly detection completed for {entity}")
                    print(f"  Total samples: {result['total_samples']}")
                    print(f"  Features used: {len(result['feature_columns'])}")
                    
                    # Print results for each model
                    for model_name, model_result in result['anomalies'].items():
                        if 'error' not in model_result:
                            print(f"  {model_name}: {model_result['count']} anomalies ({model_result['percentage']:.2f}%)")
                        else:
                            print(f"  {model_name}: ERROR - {model_result['error']}")
                    
                    break
                else:
                    print(f"✗ Anomaly detection failed for {entity}")
                    return False
        
        # Test 5: Test result saving
        print("\n5. Testing result saving...")
        detector.save_anomaly_results()
        print("✓ Results saved successfully")
        
        # Test 6: Test report generation
        print("\n6. Testing report generation...")
        report_path = detector.generate_anomaly_report()
        print(f"✓ Report generated: {report_path}")
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n" + "=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'pyod', 'anomalib'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("\n✓ All dependencies available")
        return True

if __name__ == "__main__":
    print("Starting anomaly detection tests...")
    
    # Test dependencies first
    deps_ok = test_dependencies()
    
    if deps_ok:
        # Test anomaly detection
        success = test_anomaly_detector()
        
        if success:
            print("\n🎉 ALL TESTS PASSED! Anomaly detection is ready to use.")
            print("\nNext steps:")
            print("1. Run: python notebooks/04_anomaly_detection.py")
            print("2. Check reports directory for results")
            print("3. Review generated visualizations")
        else:
            print("\n❌ TESTS FAILED! Please check the errors above.")
            sys.exit(1)
    else:
        print("\n❌ DEPENDENCIES MISSING! Please install required packages.")
        sys.exit(1) 