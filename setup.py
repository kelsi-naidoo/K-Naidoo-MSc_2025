"""
Setup script for Carbon Emissions Forecasting System v3.0.

This script helps set up the development environment and install dependencies.

Author: Kelsi Naidoo
Institution: University of Cape Town
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("Carbon Emissions Forecasting System v3.0 - Setup")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create virtual environment
    if not Path("venv").exists():
        success = run_command("python -m venv venv", "Creating virtual environment")
        if not success:
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install requirements
    success = run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")
    if not success:
        sys.exit(1)
    
    # Install Jupyter kernel
    success = run_command(f"{pip_cmd} install ipykernel", "Installing Jupyter kernel")
    if success:
        run_command(f"{pip_cmd} run python -m ipykernel install --user --name=carbon_ai_v3", 
                   "Installing Jupyter kernel")
    
    # Create necessary directories
    directories = [
        "data/processed",
        "data/sandbox", 
        "reports/figures",
        "reports/tables",
        "reports/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the data exploration notebook:")
    print("   jupyter notebook notebooks/01_data_exploration.ipynb")
    print("3. Or run the data processing pipeline:")
    print("   python src/main.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main() 