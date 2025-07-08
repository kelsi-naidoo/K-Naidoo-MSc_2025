# 🌱 Carbon Emissions AI Dashboard Guide

## Overview
The Carbon Emissions AI Dashboard is a comprehensive Streamlit application that provides an intuitive interface for analyzing carbon emissions data, comparing forecasting models, and visualizing performance metrics.

## 🚀 Quick Start

### Launch the Dashboard
```bash
# Option 1: Use the launch script (recommended)
python run_dashboard.py

# Option 2: Direct Streamlit command
streamlit run app.py

# Option 3: With custom port
streamlit run app.py --server.port 8501
```

### Access the Dashboard
- **URL**: http://localhost:8501
- **Browser**: Any modern web browser (Chrome, Firefox, Safari, Edge)

## 📋 Dashboard Features

### 1. 📁 Data Upload Page
**Purpose**: Manage and upload emissions data

**Features**:
- **Pre-loaded Entities**: View Entity A and B data that's already processed
- **File Upload**: Upload new CSV or Excel files with emissions data
- **Data Validation**: Automatic validation of uploaded files
- **Data Preview**: Preview uploaded data with key statistics

**How to Use**:
1. Navigate to "📁 Data Upload" in the sidebar
2. View pre-loaded Entity A and B data in the left panel
3. Upload new data using the file uploader in the right panel
4. Review data validation results and preview

**Supported File Formats**:
- CSV files (.csv)
- Excel files (.xlsx, .xls)

**Required Columns**:
- `emissions`: The target variable for forecasting
- `fiscalyear`: Date column (optional but recommended)

### 2. 📊 Data Overview Page
**Purpose**: Explore and visualize emissions data

**Features**:
- **Data Source Selection**: Choose between Entity A, Entity B, or uploaded data
- **Key Metrics**: Total records, emissions, averages, date ranges
- **Time Series Visualization**: Emissions over time
- **Distribution Analysis**: Histogram of emissions distribution

**How to Use**:
1. Navigate to "📊 Data Overview" in the sidebar
2. Select your data source from the dropdown
3. Review key metrics and visualizations
4. Explore data patterns and trends

### 3. 🤖 Model Performance Page
**Purpose**: Compare forecasting model accuracy

**Features**:
- **Model Comparison Tables**: RMSE, MAE, MSE, R² metrics
- **Performance Charts**: Bar charts comparing model accuracy
- **Entity-specific Results**: Separate results for each entity

**How to Use**:
1. Navigate to "🤖 Model Performance" in the sidebar
2. Review model comparison tables
3. Analyze performance charts
4. Identify the best performing model for each entity

**Available Models**:
- **ARIMA**: Time series forecasting
- **LSTM**: Deep learning neural network
- **Linear Regression**: Traditional statistical model
- **XGBoost**: Gradient boosting ensemble

### 4. 📈 Performance Analysis Page
**Purpose**: Analyze computational performance and resource usage

**Features**:
- **Computational Metrics**: Execution time, memory usage, CPU utilization
- **Performance Comparison**: Box plots and scatter plots
- **Efficiency Analysis**: Cost vs. accuracy trade-offs
- **Resource Optimization**: Recommendations for model selection

**How to Use**:
1. Navigate to "📈 Performance Analysis" in the sidebar
2. Review computational performance overview
3. Analyze performance comparison charts
4. Use efficiency analysis for model selection

**Key Metrics**:
- **Execution Time**: How long each model takes to train
- **Peak Memory**: Maximum memory usage during training
- **Peak CPU**: Maximum CPU utilization percentage
- **Efficiency Score**: Combined accuracy and computational efficiency

## 🎯 Best Practices

### Data Upload
1. **Format Your Data**: Ensure your CSV/Excel file has the required columns
2. **Clean Your Data**: Remove duplicates and handle missing values before upload
3. **Check Data Types**: Ensure dates are in a recognizable format
4. **Validate File Size**: Large files may take longer to process

### Model Selection
1. **Accuracy First**: Start with the model that has the lowest RMSE
2. **Consider Resources**: Choose models based on your computational constraints
3. **Balance Trade-offs**: Consider the accuracy vs. speed trade-off
4. **Use Performance Data**: Leverage the performance analysis for informed decisions

### Performance Optimization
1. **Speed-Critical**: Use Linear Regression or ARIMA for real-time applications
2. **Memory-Constrained**: Choose ARIMA for limited memory environments
3. **CPU-Limited**: Select XGBoost for moderate CPU usage
4. **Balanced Approach**: Use LSTM for optimal accuracy/compute ratio

## 🔧 Troubleshooting

### Common Issues

**Dashboard Won't Start**:
```bash
# Check if Streamlit is installed
pip install streamlit

# Check if you're in the right directory
ls app.py

# Check configuration
ls config/config.json
```

**No Data Available**:
- Ensure Entity A and B data files exist in the processed data directory
- Run the data cleaning pipeline first: `python notebooks/02_data_cleaning.py`

**No Model Performance Data**:
- Run the forecasting models first: `python notebooks/03_forecasting_models.py`
- Check that performance files exist in the reports directory

**Upload Errors**:
- Verify file format (CSV or Excel)
- Check that required columns are present
- Ensure file is not corrupted or empty

### Error Messages

**"Configuration file not found"**:
- Ensure `config/config.json` exists
- Check file permissions

**"Entity data not found"**:
- Run data cleaning pipeline
- Check file paths in configuration

**"No performance data found"**:
- Run forecasting models
- Check reports directory for output files

## 📊 Data Requirements

### Minimum Data Requirements
- **Emissions Column**: Required for all analyses
- **Date Column**: Recommended for time series analysis
- **Source Column**: Optional but useful for detailed analysis

### Recommended Data Structure
```csv
fiscalyear,emissions,emissions_source,property_type,sector
2020-01-01,1000.5,Electricity,Office,Commercial
2020-01-02,1200.3,Natural Gas,Retail,Commercial
...
```

### Data Quality Guidelines
- **No Missing Values**: Handle missing data before upload
- **Consistent Date Format**: Use ISO format (YYYY-MM-DD)
- **Reasonable Values**: Check for outliers and errors
- **Sufficient History**: At least 12 months of data for forecasting

## 🚀 Advanced Features

### Custom Analysis
- Upload your own data for custom analysis
- Compare multiple entities side by side
- Export results for further analysis

### Performance Monitoring
- Real-time resource usage tracking
- Computational cost analysis
- Efficiency optimization recommendations

### Model Comparison
- Side-by-side accuracy comparison
- Performance vs. computational cost analysis
- Best model recommendations for different scenarios

## 📞 Support

For technical support or questions:
1. Check the troubleshooting section above
2. Review the main README.md file
3. Run the test scripts to verify functionality
4. Check the logs for detailed error messages

## 🔄 Updates and Maintenance

### Regular Maintenance
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Clear cache: Delete `.streamlit/cache` directory if needed
- Update configuration: Modify `config/config.json` as needed

### Version Compatibility
- Python 3.8+
- Streamlit 1.44+
- Pandas 2.0+
- Plotly 5.0+

## Net Zero Pathways Tab

The Net Zero Pathways tab provides:
- A comparison of your current emissions trajectory with SBTi 1.5°C and 2°C science-based target pathways
- An interactive calculator to explore custom reduction rates and target years
- Gap analysis and recommendations

### How to Use
1. **Select the Net Zero Pathways tab** from the sidebar.
2. **Choose your entity** (Entity A or Entity B).
3. **Pick an analysis mode:**
   - *Current Trajectory vs SBTi*: See your current trend overlaid with SBTi pathways, all starting at the same base year and value.
   - *Interactive Calculator*: Use sliders to set your annual reduction rate or target net zero year and see the required pathway.
4. **Interpret the plots:**
   - All lines start at the same base year (default: 2020) and emissions value for fair comparison.
   - If the base year is missing for an entity, the dashboard uses the mean emissions and shows a warning.
5. **Review the gap analysis and recommendations** to understand your alignment with SBTi targets.

### Tips
- If your data is missing the base year, check your processed files or adjust the base year in the code.
- The dashboard is robust to different data structures for Entity A and Entity B.
- All features update live as you interact with the controls.

---

**Dashboard Version**: 1.0  
**Last Updated**: June 2025  
**Author**: Kelsi Naidoo  
**Institution**: University of Cape Town 