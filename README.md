# GCX PATH: Carbon Emissions Forecasting System v3.0

## Project Overview
AI-powered carbon emissions forecasting system for real estate sustainability data analysis. This system implements multiple machine learning models (ARIMA, LSTM, Regression, XGBoost) with comprehensive performance analysis, computational cost monitoring, and sector-based scenario testing for decarbonization planning aligned with SBTi targets.

## IEEE Standards Compliance
- **IEEE 1012-2016**: Software verification and validation
- **IEEE 829-2008**: Software test documentation
- **IEEE 730-2014**: Software quality assurance
- **IEEE 29148-2018**: Requirements engineering

## Project Objectives
1. Develop an AI-powered system to forecast carbon emissions using real estate sustainability data
2. Compare performance of various ML models (ARIMA, LSTM, Regression, XGBoost)
3. **NEW**: Implement comprehensive performance analysis and computational cost monitoring
4. **NEW**: Create intuitive Streamlit dashboard for data analysis and visualization
5. Enable sector scenario testing for decarbonization planning aligned with SBTi targets
6. Evaluate model performance using RMSE, MAE, and R² metrics
7. Maintain ethical compliance through sandbox testing and anonymized datasets

## Key Features

### 🚀 Performance Analysis & Cost Monitoring
- **Real-time resource tracking**: CPU, memory, and execution time monitoring
- **Comparative analysis**: Side-by-side performance comparison of all models
- **Cost efficiency metrics**: Computational cost vs. accuracy trade-offs
- **Scalability assessment**: Performance analysis across different dataset sizes
- **Resource optimization**: Recommendations for model selection based on constraints

### 📊 Model Performance Metrics
- **Accuracy metrics**: RMSE, MAE, R² for prediction quality
- **Computational metrics**: Execution time, peak memory, CPU usage
- **Efficiency scores**: Combined accuracy and computational efficiency
- **Cost analysis**: Memory-seconds, CPU-seconds, total computational cost

### 🎯 Smart Model Selection
- **Speed-optimized**: Fastest models for real-time applications
- **Memory-optimized**: Models for resource-constrained environments
- **CPU-optimized**: Models for CPU-limited systems
- **Balanced**: Optimal trade-off between accuracy and computational cost

### 🌱 Streamlit Dashboard
- **Interactive data upload**: Real-time file validation and preview
- **Pre-loaded entities**: Entity A and B data ready for analysis
- **Data visualization**: Interactive charts and time series plots
- **Model comparison**: Side-by-side performance analysis
- **Performance monitoring**: Computational cost analysis and recommendations

## Project Structure
```
carbon_ai_v3/
├── data/                   # Raw and processed data
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned and preprocessed data
│   └── sandbox/           # Anonymized test datasets
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_data_exploration.py
│   ├── 02_data_cleaning.py
│   ├── 03_forecasting_models.py  # Enhanced with performance monitoring
│   └── 04_results_analysis.ipynb
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # ML model implementations
│   ├── evaluation/       # Model evaluation metrics
│   └── dashboard/        # Streamlit dashboard
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
├── reports/              # Generated reports and figures
├── config/               # Configuration files
├── app.py                # NEW: Streamlit dashboard application
├── run_dashboard.py      # NEW: Dashboard launch script
├── test_dashboard.py     # NEW: Dashboard functionality test
├── demo_dashboard.py     # NEW: Dashboard demo script
├── performance_analysis.py  # NEW: Standalone performance analysis
├── test_performance_monitoring.py  # NEW: Performance monitoring tests
└── requirements.txt      # Python dependencies
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- Git

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd carbon_ai_v3
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name=carbon_ai_v3
   ```

## Usage

### 🌱 Streamlit Dashboard (NEW!)
**Launch the interactive dashboard:**
```bash
# Option 1: Use the launch script (recommended)
python run_dashboard.py

# Option 2: Direct Streamlit command
streamlit run app.py

# Option 3: With custom port
streamlit run app.py --server.port 8501
```

**Access the dashboard:**
- **URL**: http://localhost:8501
- **Browser**: Any modern web browser

**Dashboard Features:**
- 📁 **Data Upload**: Upload new data and view pre-loaded entities
- 📊 **Data Overview**: Interactive data exploration and visualization
- 🤖 **Model Performance**: Compare forecasting model accuracy
- 📈 **Performance Analysis**: Computational cost analysis and optimization

**Test Dashboard Functionality:**
```bash
# Test dashboard components
python test_dashboard.py

# Run dashboard demo
python demo_dashboard.py
```

### Data Processing Pipeline
1. **Data Exploration**: Run `notebooks/01_data_exploration.py`
2. **Data Cleaning**: Run `notebooks/02_data_cleaning.py`
3. **Model Development**: Run `notebooks/03_forecasting_models.py` (Enhanced with performance monitoring)
4. **Performance Analysis**: Run `performance_analysis.py` for detailed cost analysis

### Performance Analysis Features

#### Real-time Performance Monitoring
The forecasting system now includes comprehensive performance monitoring:
- **Execution time tracking**: Real-time measurement of model training and prediction time
- **Memory usage monitoring**: Peak and average memory consumption
- **CPU utilization tracking**: Peak and average CPU usage
- **Resource cost calculation**: Memory-seconds and CPU-seconds metrics

#### Performance Comparison Dashboard
```bash
# Run comprehensive performance analysis
python performance_analysis.py
```

**Generated Reports:**
- `performance_dashboard.png`: Visual comparison of all models
- `performance_analysis_report.txt`: Detailed performance analysis
- `{entity}_performance_metrics.csv`: Individual model performance data
- `{entity}_performance_comparison.png`: Entity-specific performance plots

#### Performance Metrics Available
- **Execution Time**: Total time for model training and prediction
- **Peak Memory**: Maximum memory usage during execution
- **Peak CPU**: Maximum CPU utilization percentage
- **Memory Efficiency**: Memory usage per unit time
- **CPU Efficiency**: CPU usage per unit time
- **Cost Efficiency Score**: Combined accuracy and computational efficiency

#### Model Selection Recommendations
The system provides intelligent recommendations based on your constraints:

```python
# Speed-optimized selection
# Recommended for real-time applications
fastest_model = "REGRESSION"  # Typically fastest

# Memory-optimized selection  
# Recommended for resource-constrained environments
memory_efficient = "ARIMA"    # Typically lowest memory

# CPU-optimized selection
# Recommended for CPU-limited systems
cpu_efficient = "XGBOOST"     # Typically moderate CPU usage

# Balanced selection
# Optimal trade-off between accuracy and computational cost
balanced = "LSTM"             # Best accuracy/compute ratio
```

### Command Line Interface
```bash
python src/main.py --config config/config.json
```

## Testing
Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

Test performance monitoring:
```bash
python test_performance_monitoring.py
```

Test dashboard functionality:
```bash
python test_dashboard.py
```

## Documentation
- **API Documentation**: `docs/api.md`
- **User Guide**: `docs/user_guide.md`
- **Technical Specifications**: `docs/technical_specs.md`
- **IEEE Compliance Report**: `docs/ieee_compliance.md`
- **Performance Analysis Guide**: `docs/performance_analysis.md`
- **Dashboard Guide**: `DASHBOARD_GUIDE.md` (NEW!)

## Ethical Considerations
- All data is anonymized and processed in sandbox environment
- No personally identifiable information (PII) is stored
- Models are validated for bias and fairness
- Results are reproducible and documented
- Performance monitoring respects privacy and doesn't collect sensitive system information

## Contributing
1. Follow IEEE coding standards
2. Write unit tests for new features
3. Update documentation
4. Submit pull requests with detailed descriptions
5. Include performance analysis for new models

## License
This project is licensed under the MIT License - see LICENSE file for details.

## Contact
- **Author**: Kelsi Naidoo
- **Institution**: University of Cape Town
- **Email**: [Your Email]
- **Project**: MSc Engineering Thesis

## Version History
- **v3.0**: Complete rebuild with IEEE standards compliance, performance analysis, and Streamlit dashboard
- **v2.0**: Previous version with basic functionality
- **v1.0**: Initial prototype 

## Net Zero Pathways Tab

The dashboard now includes a **Net Zero Pathways** tab for science-based target and custom net zero analysis. This feature allows you to:

- Compare your current emissions trajectory with SBTi 1.5°C and 2°C pathways
- Use interactive sliders to set your own annual reduction rate or target net zero year
- See all pathways start at the same base year and emissions value for fair comparison
- Robustly handles different data structures for Entity A and Entity B
- Provides gap analysis and recommendations

**How it works:**
- The base year (default: 2020) is used for all pathway calculations. If data for the base year is missing, the dashboard falls back to the mean emissions value and displays a warning.
- The tab is accessible from the sidebar. Select your entity and analysis mode (Current Trajectory vs SBTi or Interactive Calculator).
- All plots and tables update live based on your selections.

See the dashboard for more details and interactive features.

--- 
