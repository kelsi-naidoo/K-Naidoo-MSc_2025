# Quick Start Guide - Carbon Emissions Forecasting System v3.0

## Prerequisites
- Python 3.8 or higher
- Git

## Step 1: Setup Environment

### Option A: Automated Setup (Recommended)
```bash
# Run the setup script
python setup.py
```

### Option B: Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Jupyter kernel
python -m ipykernel install --user --name=carbon_ai_v3
```

## Step 2: Data Exploration

Start with the data exploration notebook to understand your data:

```bash
# Launch Jupyter
jupyter notebook notebooks/01_data_exploration.ipynb
```

This notebook will:
- Load your emissions data
- Display data structure and quality
- Generate initial insights
- Save exploration results

## Step 3: Data Cleaning

Run the data cleaning notebook:

```bash
jupyter notebook notebooks/02_data_cleaning.ipynb
```

This notebook will:
- Clean and standardize your data
- Handle missing values and outliers
- Create temporal and sector features
- Validate data quality
- Prepare data for machine learning

## Step 4: Command Line Processing (Alternative)

If you prefer command line processing:

```bash
# Run complete pipeline
python src/main.py

# Or run specific steps
python src/main.py --step load    # Load data only
python src/main.py --step clean   # Clean data only
python src/main.py --step validate # Validate data only
```

## Step 5: View Results

Check the generated files:

- **Processed Data**: `data/processed/`
- **Reports**: `reports/`
- **Figures**: `reports/figures/`
- **Logs**: `reports/logs/`

## Project Structure

```
carbon_ai_v3/
├── data/                   # Your data files
│   ├── raw/               # Original data (copied from parent)
│   └── processed/         # Cleaned and processed data
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_data_cleaning.ipynb
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # ML models (to be implemented)
│   └── main.py           # Main pipeline script
├── config/               # Configuration files
├── reports/              # Generated outputs
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the virtual environment
2. **File Not Found**: Check that your data files are in `data/raw/`
3. **Memory Issues**: Consider processing entities one at a time

### Getting Help

1. Check the logs in `reports/logs/`
2. Review the validation reports in `data/processed/`
3. Run the unit tests: `python -m pytest tests/`

## Next Steps

After completing data exploration and cleaning:

1. **Model Development**: Implement forecasting models
2. **Evaluation**: Compare model performance
3. **Dashboard**: Create interactive Streamlit dashboard
4. **Documentation**: Complete technical documentation

## IEEE Standards Compliance

This project follows IEEE standards:
- **IEEE 1012-2016**: Software verification and validation
- **IEEE 829-2008**: Software test documentation
- **IEEE 730-2014**: Software quality assurance
- **IEEE 29148-2018**: Requirements engineering

All processes are documented, tested, and reproducible for academic rigor. 