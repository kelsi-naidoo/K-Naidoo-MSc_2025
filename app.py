"""
Carbon Emissions Forecasting Dashboard
Streamlit Application for AI-powered Carbon Emissions Analysis

Author: Kelsi Naidoo
Institution: University of Cape Town
Date: June 2025

Features:
- Data upload and validation
- Pre-loaded Entity A and B data
- Model performance comparison
- Performance analysis visualization
- Clean, intuitive interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
from datetime import datetime, timedelta
import warnings
import io
import base64
import random
from scipy import stats
import glob
import os
import importlib.util

# Import net zero pathway functions
sys.path.append('notebooks')
spec = importlib.util.spec_from_file_location("net_zero_pathway", "notebooks/05_net_zero_pathway.py")
net_zero_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net_zero_module)
NetZeroPathway = net_zero_module.NetZeroPathway
interactive_pathway_analysis = net_zero_module.interactive_pathway_analysis
calculate_current_trajectory = net_zero_module.calculate_current_trajectory

# Suppress warnings
warnings.filterwarnings('ignore')

# Add src to path for importing our modules
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="GCX PATH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration file."""
    try:
        with open('config/config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config/config.json exists.")
        return None

# Load pre-existing data
@st.cache_data
def load_entity_data():
    """Load pre-existing Entity A and B data."""
    config = load_config()
    if not config:
        return None, None
    
    processed_dir = Path(config['data']['processed_dir'])
    
    # Load Entity A data
    entity_a_file = processed_dir / 'cleaned_EntityA.csv'
    entity_a_data = None
    if entity_a_file.exists():
        try:
            entity_a_data = pd.read_csv(entity_a_file)
            # Convert fiscalyear to datetime if it exists
            if 'fiscalyear' in entity_a_data.columns:
                entity_a_data['fiscalyear'] = pd.to_datetime(entity_a_data['fiscalyear'])
        except Exception as e:
            st.error(f"Error loading Entity A data: {e}")
    
    # Load Entity B data
    entity_b_file = processed_dir / 'cleaned_EntityB.csv'
    entity_b_data = None
    if entity_b_file.exists():
        try:
            entity_b_data = pd.read_csv(entity_b_file)
            # Convert fiscalyear to datetime if it exists
            if 'fiscalyear' in entity_b_data.columns:
                entity_b_data['fiscalyear'] = pd.to_datetime(entity_b_data['fiscalyear'])
        except Exception as e:
            st.error(f"Error loading Entity B data: {e}")
    
    return entity_a_data, entity_b_data

def calculate_entity_emissions(entity_a, entity_b):
    """Calculate emissions summary for both entities"""
    if entity_a is None or entity_b is None:
        return None

    summary = {}

    for name, data in [("Entity A", entity_a), ("Entity B", entity_b)]:
        # Calculate total emissions
        total_emissions = data['emissions'].sum()

        # Calculate emissions by year
        yearly_emissions = data.groupby('fiscalyear')['emissions'].sum().reset_index()

        # Calculate sector distribution
        if name == "Entity A":
            sector_col = 'sector'
        else:
            sector_col = 'property_type' if 'property_type' in data.columns else 'primary_category'

        sector_distribution = data[sector_col].value_counts() if sector_col in data.columns else None

        summary[name] = {
            'total_emissions': total_emissions,
            'yearly_emissions': yearly_emissions,
            'sector_distribution': sector_distribution,
            'data': data
        }

    return summary

# Data validation function
def validate_uploaded_file(uploaded_file):
    """Validate uploaded file format and content."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return False, f"Invalid file format. Please upload {', '.join(allowed_extensions)} files only."
    
    try:
        # Try to read the file
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Check if file is empty
        if df.empty:
            return False, "Uploaded file is empty."
        
        # Check for required columns (basic check)
        required_columns = ['emissions']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        return True, f"File validated successfully. Shape: {df.shape}"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

# File upload function
def upload_file():
    """Handle file upload with validation."""
    
    # Template download section
    st.markdown("### 📋 Download Template")
    st.write("Download a template CSV file to see the required format:")
    
    template_df = create_template_csv()
    
    # Display template preview
    st.write("**Template Preview:**")
    st.dataframe(template_df.head(), use_container_width=True)
    
    # Download button
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv,
        file_name="emissions_template.csv",
        mime="text/csv",
        help="Download a template CSV file with the correct format"
    )
    
    st.markdown("---")
    
    # Format requirements
    with st.expander("📋 Format Requirements", expanded=False):
        st.markdown("""
        **Required Columns:**
        - `emissions` (numeric) - Carbon emissions values
        - `fiscalyear` (date) - Date of emissions (recommended)
        
        **Optional Columns:**
        - `emissions_source` (text) - Source of emissions
        - `property_type` (text) - Type of property
        - `sector` (text) - Business sector
        - `location` (text) - Geographic location
        - `building_id` (text) - Building identifier
        
        **Accepted Formats:**
        - CSV files (.csv) - Recommended
        - Excel files (.xlsx, .xls)
        
        **Date Formats:**
        - ISO: `2023-01-15` (recommended)
        - US: `01/15/2023`
        - European: `15/01/2023`
        """)
    
    st.markdown("### 📁 Upload New Data")
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files containing emissions data"
    )
    
    if uploaded_file is not None:
        # Validate the uploaded file
        is_valid, message = validate_uploaded_file(uploaded_file)
        
        if is_valid:
            st.success(message)
            
            # Read the file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Display file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Show data preview
                st.markdown("#### 📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Show data info
                with st.expander("📋 Data Information"):
                    st.write("**Column Information:**")
                    st.write(df.info())
                    
                    st.write("**Missing Values:**")
                    missing_data = df.isnull().sum()
                    if missing_data.sum() > 0:
                        st.write(missing_data[missing_data > 0])
                    else:
                        st.write("No missing values found.")
                
                # Store in session state
                st.session_state['uploaded_data'] = df
                st.session_state['uploaded_filename'] = uploaded_file.name
                
                return df
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
        else:
            st.error(message)
            return None
    
    return None

# Display pre-loaded data
def display_preloaded_data(entity_a_data, entity_b_data):
    """Display pre-loaded Entity A and B data."""
    st.markdown("### 🏢 Pre-loaded Entity Data")
    
    # Create tabs for each entity
    tab1, tab2 = st.tabs(["Entity A", "Entity B"])
    
    with tab1:
        if entity_a_data is not None:
            st.success("✅ Entity A data loaded successfully")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(entity_a_data))
            with col2:
                st.metric("Date Range", f"{entity_a_data['fiscalyear'].min().strftime('%Y-%m-%d')} to {entity_a_data['fiscalyear'].max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Total Emissions", f"{entity_a_data['emissions'].sum():,.0f}")
            
            # Data preview
            st.markdown("#### 📊 Entity A Data Preview")
            st.dataframe(entity_a_data.head(), use_container_width=True)
            
            # Basic statistics
            with st.expander("📈 Entity A Statistics"):
                st.write("**Emissions Statistics:**")
                st.write(entity_a_data['emissions'].describe())
                
                if 'emissions_source' in entity_a_data.columns:
                    st.write("**Emissions by Source:**")
                    source_counts = entity_a_data['emissions_source'].value_counts()
                    st.bar_chart(source_counts)
        else:
            st.warning("⚠️ Entity A data not found. Please ensure the data file exists.")
    
    with tab2:
        if entity_b_data is not None:
            st.success("✅ Entity B data loaded successfully")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(entity_b_data))
            with col2:
                st.metric("Date Range", f"{entity_b_data['fiscalyear'].min().strftime('%Y-%m-%d')} to {entity_b_data['fiscalyear'].max().strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Total Emissions", f"{entity_b_data['emissions'].sum():,.0f}")
            
            # Data preview
            st.markdown("#### 📊 Entity B Data Preview")
            st.dataframe(entity_b_data.head(), use_container_width=True)
            
            # Basic statistics
            with st.expander("📈 Entity B Statistics"):
                st.write("**Emissions Statistics:**")
                st.write(entity_b_data['emissions'].describe())
                
                if 'emissions_source' in entity_b_data.columns:
                    st.write("**Emissions by Source:**")
                    source_counts = entity_b_data['emissions_source'].value_counts()
                    st.bar_chart(source_counts)
        else:
            st.warning("⚠️ Entity B data not found. Please ensure the data file exists.")

def create_template_csv():
    """Create a simple CSV template for download."""
    template_data = {
        'fiscalyear': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'emissions': [1250.5, 1180.3, 1320.7, 1100.2, 1450.8],
        'emissions_source': ['Electricity', 'Natural Gas', 'Diesel', 'Electricity', 'Natural Gas'],
        'property_type': ['Office', 'Retail', 'Industrial', 'Office', 'Retail'],
        'sector': ['Commercial', 'Commercial', 'Industrial', 'Commercial', 'Commercial']
    }
    return pd.DataFrame(template_data)

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def load_latest_anomaly_data():
    """Load the latest anomaly detection results."""
    try:
        # Look for the latest anomaly summary file
        summary_files = glob.glob('reports/anomaly_summary_*.json')
        if not summary_files:
            return None, None, []
        
        # Get the most recent file
        latest_file = max(summary_files, key=os.path.getctime)
        
        # Extract timestamp from filename
        timestamp = latest_file.split('_')[-1].replace('.json', '')
        
        # Load summary data
        with open(latest_file, 'r') as f:
            summary_data = json.load(f)
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Look for detailed analysis files
        detailed_files = glob.glob('reports/anomaly_detailed_*.md')
        
        return summary_df, timestamp, detailed_files
        
    except Exception as e:
        st.error(f"Error loading anomaly data: {e}")
        return None, None, []

def load_anomaly_reports(entity_name):
    """Load anomaly detection reports for a specific entity."""
    try:
        # Look for entity-specific anomaly reports
        report_files = glob.glob(f'reports/anomaly_{entity_name.replace(" ", "")}_*.json')
        if not report_files:
            return None
        
        # Get the most recent file
        latest_file = max(report_files, key=os.path.getctime)
        
        # Load report data
        with open(latest_file, 'r') as f:
            report_data = json.load(f)
        
        return report_data
        
    except Exception as e:
        st.error(f"Error loading anomaly reports for {entity_name}: {e}")
        return None

def parse_detailed_analysis(md_file_path):
    """Parse detailed analysis markdown file."""
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract entity name from filename
        filename = os.path.basename(md_file_path)
        entity = filename.split('_')[2] if len(filename.split('_')) > 2 else "Unknown"
        
        # Parse the markdown content
        analysis = {
            'entity': entity,
            'emission_sources': {},
            'property_analysis': {},
            'temporal_analysis': {}
        }
        
        # Simple parsing - look for sections
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('##'):
                current_section = line.replace('#', '').strip().lower()
            elif line.startswith('###'):
                subsection = line.replace('#', '').strip().lower()
                current_section = f"{current_section}_{subsection}"
            elif line.startswith('-') or line.startswith('*'):
                # Parse list items
                if 'emission' in current_section and 'source' in current_section:
                    # Parse emission source data
                    parts = line.replace('-', '').replace('*', '').strip().split(':')
                    if len(parts) == 2:
                        source = parts[0].strip()
                        count = int(parts[1].strip().split()[0])
                        if 'z-score' not in analysis['emission_sources']:
                            analysis['emission_sources']['z-score'] = {}
                        analysis['emission_sources']['z-score'][source] = count
                elif 'property' in current_section:
                    # Parse property data
                    parts = line.replace('-', '').replace('*', '').strip().split(':')
                    if len(parts) == 2:
                        property_name = parts[0].strip()
                        count = int(parts[1].strip().split()[0])
                        if 'z-score' not in analysis['property_analysis']:
                            analysis['property_analysis']['z-score'] = {}
                        if 'property_name' not in analysis['property_analysis']['z-score']:
                            analysis['property_analysis']['z-score']['property_name'] = {}
                        analysis['property_analysis']['z-score']['property_name'][property_name] = count
        
        return analysis
        
    except Exception as e:
        st.error(f"Error parsing detailed analysis: {e}")
        return None

def anomaly_detection_tab(entity_a_data, entity_b_data):
    """Anomaly Detection Tab"""
    st.header("🔍 Anomaly Detection")
    st.markdown("---")
    
    # Entity selection
    selected_entity = st.selectbox(
        "Select Entity for Anomaly Analysis",
        ["Entity A", "Entity B"],
        help="Choose which entity to analyze for anomalies"
    )
    
    # Get data for selected entity
    if selected_entity == "Entity A":
        data = entity_a_data
    else:
        data = entity_b_data
    
    if data is None:
        st.error("❌ No data available for analysis.")
        return
    
    # Load anomaly reports
    anomaly_data = load_anomaly_reports(selected_entity)
    
    if anomaly_data:
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Records",
                f"{anomaly_data.get('total_records', 0):,}",
                help="Total number of records analyzed"
            )
        
        with col2:
            anomaly_count = anomaly_data.get('anomaly_count', 0)
            total_records = anomaly_data.get('total_records', 1)
            anomaly_percentage = (anomaly_count / total_records) * 100 if total_records > 0 else 0
            
            st.metric(
                "Anomalies Detected",
                f"{anomaly_count:,}",
                delta=f"{anomaly_percentage:.1f}%",
                delta_color="inverse" if anomaly_percentage > 5 else "normal",
                help="Number and percentage of anomalies detected"
            )
        
        with col3:
            st.metric(
                "Detection Method",
                anomaly_data.get('method', 'Z-Score'),
                help="Statistical method used for anomaly detection"
            )
        
        # Display anomaly details
        if 'property_details' in anomaly_data and anomaly_data['property_details']:
            st.subheader("🏢 Anomalies by Property")
            
            # Create property summary
            property_summary = []
            for prop, details in anomaly_data['property_details'].items():
                property_summary.append({
                    'Property': prop,
                    'Anomaly Count': details.get('count', 0),
                    'Average Emissions': f"{details.get('avg_emissions', 0):.2f}",
                    'Max Emissions': f"{details.get('max_emissions', 0):.2f}"
                })
            
            if property_summary:
                prop_df = pd.DataFrame(property_summary)
                prop_df = prop_df[prop_df['Anomaly Count'] > 0].sort_values('Anomaly Count', ascending=False)
                
                if not prop_df.empty:
                    st.dataframe(prop_df, use_container_width=True, hide_index=True)
                else:
                    st.info("✅ No property-specific anomalies found.")
        
        if 'source_details' in anomaly_data and anomaly_data['source_details']:
            st.subheader("⚡ Anomalies by Emission Source")
            
            # Create source summary
            source_summary = []
            for source, details in anomaly_data['source_details'].items():
                source_summary.append({
                    'Emission Source': source,
                    'Anomaly Count': details.get('count', 0),
                    'Average Emissions': f"{details.get('avg_emissions', 0):.2f}",
                    'Max Emissions': f"{details.get('max_emissions', 0):.2f}"
                })
            
            if source_summary:
                source_df = pd.DataFrame(source_summary)
                source_df = source_df[source_df['Anomaly Count'] > 0].sort_values('Anomaly Count', ascending=False)
                
                if not source_df.empty:
                    st.dataframe(source_df, use_container_width=True, hide_index=True)
                else:
                    st.info("✅ No source-specific anomalies found.")
        
        # Display temporal analysis if available
        if 'temporal_analysis' in anomaly_data and anomaly_data['temporal_analysis']:
            st.subheader("📅 Temporal Analysis")
            
            temporal_data = anomaly_data['temporal_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'yearly_trend' in temporal_data:
                    st.markdown("**Yearly Anomaly Trend**")
                    yearly_df = pd.DataFrame(temporal_data['yearly_trend'])
                    st.dataframe(yearly_df, use_container_width=True, hide_index=True)
            
            with col2:
                if 'monthly_pattern' in temporal_data:
                    st.markdown("**Monthly Pattern**")
                    monthly_df = pd.DataFrame(temporal_data['monthly_pattern'])
                    st.dataframe(monthly_df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("⚠️ No anomaly detection reports found.")
        st.info("💡 Run the anomaly detection notebook to generate reports.")

def net_zero_pathways_tab():
    """Net Zero Pathways Analysis Tab"""
    st.header("🌱 Net Zero Pathways Analysis")
    st.markdown("---")
    
    # Load data
    entity_a, entity_b = load_entity_data()
    if entity_a is None or entity_b is None:
        st.error("❌ Data not available. Please run data processing first.")
        return
    
    # Calculate emissions summary
    emissions_summary = calculate_entity_emissions(entity_a, entity_b)
    if emissions_summary is None:
        return
    
    # Sidebar for controls
    st.sidebar.header("🎛️ Pathway Controls")
    
    # Entity selection
    selected_entity = st.sidebar.selectbox(
        "Select Entity",
        ["Entity A", "Entity B"],
        help="Choose which entity to analyze"
    )
    
    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Current Trajectory vs SBTi", "Interactive Calculator"],
        help="Choose analysis type"
    )
    
    if analysis_mode == "Current Trajectory vs SBTi":
        show_current_trajectory_analysis(emissions_summary, selected_entity)
    else:
        show_interactive_calculator(emissions_summary, selected_entity)

def show_current_trajectory_analysis(emissions_summary, selected_entity):
    """Show current trajectory analysis with SBTi comparison"""
    
    # Get entity data
    entity_data = emissions_summary[selected_entity]['data']
    yearly_emissions = emissions_summary[selected_entity]['yearly_emissions']
    base_year = 2020

    # Robust base_emissions extraction
    base_emissions = None
    if 'fiscalyear' in yearly_emissions.columns:
        # If fiscalyear is datetime, get year
        if np.issubdtype(yearly_emissions['fiscalyear'].dtype, np.datetime64):
            base_emissions = yearly_emissions[yearly_emissions['fiscalyear'].dt.year == base_year]['emissions'].sum()
        else:
            base_emissions = yearly_emissions[yearly_emissions['fiscalyear'] == base_year]['emissions'].sum()
    elif 'Year' in yearly_emissions.columns:
        base_emissions = yearly_emissions[yearly_emissions['Year'] == base_year]['emissions'].sum()
    else:
        st.warning(f"Could not find a valid year column for {selected_entity}. Using mean emissions as fallback.")
        base_emissions = yearly_emissions['emissions'].mean()

    # If still zero or NaN, fallback to mean
    if base_emissions == 0 or pd.isna(base_emissions):
        st.warning(f"No emissions found for {base_year} for {selected_entity}. Using mean emissions as fallback.")
        base_emissions = yearly_emissions['emissions'].mean()

    # Calculate current trajectory
    current_trajectory = calculate_current_trajectory(entity_data, selected_entity)
    
    # Calculate SBTi pathways
    pathway_calc = NetZeroPathway()
    pathway_1_5c = pathway_calc.calculate_pathway(base_emissions, '1.5C')
    pathway_2c = pathway_calc.calculate_pathway(base_emissions, '2C')
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            f"Base Emissions ({base_year})",
            f"{base_emissions:,.0f} tCO2e",
            help="Emissions for the base year used in all pathways"
        )
    
    with col2:
        current_trend = current_trajectory['trend_description']
        st.metric(
            "Current Trend",
            current_trend,
            help="Annual change in emissions"
        )
    
    with col3:
        current_nz_year = current_trajectory['net_zero_year']
        sbti_nz_year = pathway_calc.calculate_net_zero_timeline(base_emissions)['1.5C']['net_zero_year']
        gap = current_nz_year - sbti_nz_year if current_nz_year else None
        
        if gap and gap < 0:
            st.metric(
                "Net Zero Gap",
                f"{abs(gap)} years ahead",
                delta="✅ Ahead of SBTi",
                delta_color="normal"
            )
        elif gap and gap > 0:
            st.metric(
                "Net Zero Gap",
                f"{gap} years behind",
                delta="⚠️ Behind SBTi",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Net Zero Gap",
                "On track",
                delta="✅ On target",
                delta_color="normal"
            )
    
    with col4:
        # Calculate reduction needed by 2030
        current_2030 = current_trajectory['trajectory'][current_trajectory['trajectory']['Year'] == 2030]['Emissions'].iloc[0]
        sbti_2030 = pathway_1_5c[pathway_1_5c['Year'] == 2030]['Emissions'].iloc[0]
        reduction_needed = ((current_2030 - sbti_2030) / current_2030) * 100 if current_2030 > 0 else 0
        
        if reduction_needed > 0:
            st.metric(
                "Additional Reduction Needed (2030)",
                f"{reduction_needed:.1f}%",
                delta="⚠️ More reduction needed",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Additional Reduction Needed (2030)",
                "0%",
                delta="✅ On track",
                delta_color="normal"
            )
    
    # Create pathway comparison chart
    st.subheader("📈 Pathway Comparison")
    
    # Prepare data for plotting
    years = range(base_year, 2051)
    
    # Current trajectory
    current_df = current_trajectory['trajectory']
    current_df = current_df[(current_df['Year'] >= base_year) & (current_df['Year'] <= 2050)]
    
    # SBTi pathways
    sbti_1_5c_df = pathway_1_5c[(pathway_1_5c['Year'] >= base_year) & (pathway_1_5c['Year'] <= 2050)]
    sbti_2c_df = pathway_2c[(pathway_2c['Year'] >= base_year) & (pathway_2c['Year'] <= 2050)]
    
    # Create plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=current_df['Year'],
        y=current_df['Emissions'],
        mode='lines',
        name='Current Trend',
        line=dict(color='#FF8C00', width=3),
        hovertemplate='Year: %{x}<br>Emissions: %{y:,.0f} tCO2e<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=sbti_1_5c_df['Year'],
        y=sbti_1_5c_df['Emissions'],
        mode='lines',
        name='SBTi 1.5°C',
        line=dict(color='#2E8B57', width=2),
        hovertemplate='Year: %{x}<br>Emissions: %{y:,.0f} tCO2e<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=sbti_2c_df['Year'],
        y=sbti_2c_df['Emissions'],
        mode='lines',
        name='SBTi 2°C',
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        hovertemplate='Year: %{x}<br>Emissions: %{y:,.0f} tCO2e<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Emissions Pathways - {selected_entity}",
        xaxis_title="Year",
        yaxis_title="Emissions (tCO2e)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gap analysis table
    st.subheader("📊 Gap Analysis")
    
    gap_data = {
        'Metric': [
            'Current Net Zero Year',
            'SBTi 1.5°C Net Zero Year',
            'SBTi 2°C Net Zero Year',
            'Gap vs 1.5°C',
            'Gap vs 2°C',
            'Current Emissions 2030',
            'SBTi 1.5°C Emissions 2030',
            'Additional Reduction Needed 2030'
        ],
        'Value': [
            f"{current_nz_year}" if current_nz_year else "Never",
            f"{pathway_calc.calculate_net_zero_timeline(base_emissions)['1.5C']['net_zero_year']}",
            f"{pathway_calc.calculate_net_zero_timeline(base_emissions)['2C']['net_zero_year']}",
            f"{gap} years" if gap else "N/A",
            f"{current_nz_year - pathway_calc.calculate_net_zero_timeline(base_emissions)['2C']['net_zero_year']} years" if current_nz_year else "N/A",
            f"{current_2030:,.0f} tCO2e",
            f"{sbti_2030:,.0f} tCO2e",
            f"{reduction_needed:.1f}%" if reduction_needed > 0 else "0% (on track)"
        ]
    }
    
    gap_df = pd.DataFrame(gap_data)
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

def show_interactive_calculator(emissions_summary, selected_entity):
    """Show interactive calculator with sliders"""
    
    # Get entity data
    base_emissions = emissions_summary[selected_entity]['yearly_emissions']['emissions'].mean()
    
    st.subheader("🎯 Interactive Net Zero Calculator")
    
    # Two-column layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 Calculate Net Zero Year")
        st.markdown("Adjust the reduction rate to see when you'll reach net zero:")
        
        reduction_rate = st.slider(
            "Annual Reduction Rate (%)",
            min_value=1.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Percentage reduction in emissions per year"
        )
        
        # Calculate net zero year
        pathway_calc = NetZeroPathway()
        net_zero_year = pathway_calc.calculate_net_zero_year(base_emissions, reduction_rate/100)
        
        if net_zero_year:
            st.success(f"🎯 **Net Zero Year: {net_zero_year}**")
            years_to_net_zero = net_zero_year - 2020
            st.info(f"This pathway will take {years_to_net_zero} years to reach net zero.")
        else:
            st.warning("⚠️ This reduction rate won't reach net zero by 2100.")
    
    with col2:
        st.markdown("### 📅 Calculate Required Reduction Rate")
        st.markdown("Set a target year to see the required reduction rate:")
        
        target_year = st.slider(
            "Target Net Zero Year",
            min_value=2025,
            max_value=2100,
            value=2050,
            step=1,
            help="Year by which you want to reach net zero"
        )
        
        # Calculate required reduction rate
        required_rate = pathway_calc.calculate_required_reduction_rate(base_emissions, target_year)
        
        if required_rate <= 1.0:
            st.success(f"🎯 **Required Reduction Rate: {required_rate*100:.1f}%**")
            st.info(f"You need to reduce emissions by {required_rate*100:.1f}% annually to reach net zero by {target_year}.")
        else:
            st.error("❌ **Impossible Target**")
            st.warning(f"Reaching net zero by {target_year} is not possible with current emissions levels.")
    
    # Comparison with SBTi
    st.subheader("🔬 Comparison with SBTi Pathways")
    
    # Calculate SBTi pathways
    sbti_result = interactive_pathway_analysis(base_emissions)
    sbti_1_5c_year = sbti_result['net_zero_timeline']['1.5C']['net_zero_year']
    sbti_2c_year = sbti_result['net_zero_timeline']['2C']['net_zero_year']
    
    # Create comparison chart
    if net_zero_year:
        comparison_data = {
            'Pathway': ['Your Pathway', 'SBTi 1.5°C', 'SBTi 2°C'],
            'Net Zero Year': [net_zero_year, sbti_1_5c_year, sbti_2c_year],
            'Reduction Rate (%)': [reduction_rate, 7.4, 2.5]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create bar chart
        fig = px.bar(
            comparison_df,
            x='Pathway',
            y='Net Zero Year',
            color='Pathway',
            title=f"Net Zero Year Comparison - {selected_entity}",
            color_discrete_map={
                'Your Pathway': '#FF8C00',
                'SBTi 1.5°C': '#2E8B57',
                'SBTi 2°C': '#FF6B6B'
            }
        )
        
        fig.update_layout(
            yaxis_title="Net Zero Year",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("### 📋 Pathway Summary")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if net_zero_year < sbti_1_5c_year:
            st.success("🎉 **Excellent!** Your pathway is more aggressive than the 1.5°C SBTi target.")
        elif net_zero_year < sbti_2c_year:
            st.info("👍 **Good progress!** Your pathway is more aggressive than the 2°C SBTi target.")
        else:
            st.warning("⚠️ **Consider increasing ambition** to align with SBTi targets.")

# Main dashboard function
def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">GCX PATH</h1>', unsafe_allow_html=True)
    st.markdown("AI-powered carbon emissions forecasting and analysis system")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📁 Data Upload", "📊 Data Overview", "🔍 Anomaly Detection", "🤖 Model Performance", "📈 Performance Analysis", "🌱 Net Zero Pathways"]
    )
    
    # Load data
    entity_a_data, entity_b_data = load_entity_data()
    
    # Page routing
    if page == "📁 Data Upload":
        st.markdown('<h2 class="sub-header">📁 Data Management</h2>', unsafe_allow_html=True)
        
        # Template and upload section
        uploaded_data = upload_file()
        
        if uploaded_data is not None:
            st.markdown("### ✅ Upload Summary")
            st.success(f"Successfully uploaded: **{st.session_state.get('uploaded_filename', 'Unknown file')}**")
            
            # Store in session state for other pages
            st.session_state['current_data'] = uploaded_data
        
        st.markdown("---")
        
        # Pre-loaded data section
        st.markdown("### 🏢 Pre-loaded Entities")
        display_preloaded_data(entity_a_data, entity_b_data)
    
    elif page == "📊 Data Overview":
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        # Select data source
        data_sources = []
        if entity_a_data is not None:
            data_sources.append("Entity A")
        if entity_b_data is not None:
            data_sources.append("Entity B")
        if 'current_data' in st.session_state:
            data_sources.append("Uploaded Data")
        
        if not data_sources:
            st.warning("No data available. Please upload data or ensure Entity A/B files exist.")
            return
        
        selected_source = st.selectbox("Select data source:", data_sources)
        
        # Get selected data
        if selected_source == "Entity A":
            data = entity_a_data
            title = "Entity A"
        elif selected_source == "Entity B":
            data = entity_b_data
            title = "Entity B"
        else:
            data = st.session_state['current_data']
            title = "Uploaded Data"
        
        if data is not None:
            st.markdown(f"### 📈 {title} Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Total Emissions", f"{data['emissions'].sum():,.0f}")
            with col3:
                st.metric("Average Emissions", f"{data['emissions'].mean():,.0f}")
            with col4:
                st.metric("Date Range", f"{data['fiscalyear'].min().strftime('%Y-%m-%d')} to {data['fiscalyear'].max().strftime('%Y-%m-%d')}")
            
            # Time series plot
            st.markdown("#### 📈 Emissions Over Time")
            if 'fiscalyear' in data.columns:
                fig = px.line(data, x='fiscalyear', y='emissions', title=f"{title} Emissions Timeline")
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plot
            st.markdown("#### 📊 Emissions Distribution")
            fig = px.histogram(data, x='emissions', title=f"{title} Emissions Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sector distribution
            if 'sector' in data.columns:
                st.markdown("#### 🏢 Sector Distribution")
                
                # Calculate sector distribution
                sector_counts = data['sector'].value_counts()
                
                # Create pie chart
                fig = px.pie(
                    values=sector_counts.values, 
                    names=sector_counts.index, 
                    title=f"{title} - Sector Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show sector statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sector Breakdown:**")
                    for sector, count in sector_counts.items():
                        percentage = (count / len(data)) * 100
                        st.write(f"• {sector}: {count:,} records ({percentage:.1f}%)")
                
                with col2:
                    st.write("**Emissions by Sector:**")
                    sector_emissions = data.groupby('sector')['emissions'].sum().sort_values(ascending=False)
                    for sector, emissions in sector_emissions.items():
                        st.write(f"• {sector}: {emissions:,.0f}")
            else:
                st.info("ℹ️ No sector information available in this dataset.")
    
    elif page == "🤖 Model Performance":
        st.markdown('<h2 class="sub-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
        
        # Check if performance data exists
        reports_dir = Path("reports")
        performance_files = list(reports_dir.glob("*_model_comparison.csv"))
        
        if not performance_files:
            st.info("No model performance data found. Please run the forecasting models first.")
            st.markdown("""
            To generate performance data:
            1. Run `python notebooks/03_forecasting_models.py`
            2. This will create performance comparison files
            3. Refresh this page to view results
            """)
            return
        
        # Load and display performance data
        st.markdown("### 📊 Model Comparison Results")
        
        for file in performance_files:
            entity_name = file.stem.replace('_model_comparison', '')
            st.markdown(f"#### {entity_name}")
            
            try:
                df = pd.read_csv(file)
                st.dataframe(df, use_container_width=True)
                
                # Create performance comparison chart
                fig = px.bar(df, x='Model', y='RMSE', title=f"{entity_name} - Model RMSE Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
    
    elif page == "📈 Performance Analysis":
        st.markdown('<h2 class="sub-header">📈 Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Check if performance analysis files exist
        reports_dir = Path("reports")
        performance_files = list(reports_dir.glob("*_performance_metrics.csv"))
        
        if not performance_files:
            st.info("No performance analysis data found. Please run the forecasting models first.")
            return
        
        # Load all performance data
        all_performance = []
        for file in performance_files:
            try:
                df = pd.read_csv(file)
                all_performance.append(df)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
        
        if all_performance:
            performance_df = pd.concat(all_performance, ignore_index=True)
            
            st.markdown("### 🚀 Computational Performance Overview")
            
            # Key performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                fastest = performance_df.loc[performance_df['Execution_Time_Seconds'].idxmin()]
                st.metric("Fastest Model", f"{fastest['Model']} ({fastest['Entity']})", f"{fastest['Execution_Time_Seconds']:.2f}s")
            
            with col2:
                most_memory = performance_df.loc[performance_df['Peak_Memory_MB'].idxmin()]
                st.metric("Most Memory Efficient", f"{most_memory['Model']} ({most_memory['Entity']})", f"{most_memory['Peak_Memory_MB']:.1f}MB")
            
            with col3:
                most_cpu = performance_df.loc[performance_df['Peak_CPU_Percent'].idxmin()]
                st.metric("Most CPU Efficient", f"{most_cpu['Model']} ({most_cpu['Entity']})", f"{most_cpu['Peak_CPU_Percent']:.1f}%")
            
            # Performance comparison charts
            st.markdown("### 📊 Performance Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(performance_df, x='Model', y='Execution_Time_Seconds', 
                           title="Execution Time Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(performance_df, x='Model', y='Peak_Memory_MB', 
                           title="Memory Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Efficiency analysis
            st.markdown("### ⚡ Efficiency Analysis")
            
            # Calculate efficiency scores
            performance_df['Efficiency_Score'] = (
                performance_df['Execution_Time_Seconds'] * 
                performance_df['Peak_Memory_MB'] * 
                performance_df['Peak_CPU_Percent']
            )
            
            fig = px.scatter(performance_df, x='Execution_Time_Seconds', y='Peak_Memory_MB',
                           size='Peak_CPU_Percent', color='Model', hover_data=['Entity'],
                           title="Performance Efficiency: Time vs Memory (Size = CPU Usage)")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "🔍 Anomaly Detection":
        anomaly_detection_tab(entity_a_data, entity_b_data)

    elif page == "🌱 Net Zero Pathways":
        net_zero_pathways_tab()

# Run the app
if __name__ == "__main__":
    main() 