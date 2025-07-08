"""
SBTi and Sector-Aligned Net Zero Pathway Analysis
================================================

This notebook implements Science Based Targets initiative (SBTi) compliant
emissions reduction pathways and sector-specific net zero analysis.

Features:
- SBTi 1.5°C and 2°C pathway calculations
- Sector-specific reduction targets
- Net zero timeline analysis
- Carbon budget calculations
- Pathway visualization and reporting

Author: MSc Student
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetZeroPathway:
    """
    SBTi and sector-aligned net zero pathway calculator
    """
    
    def __init__(self, base_year=2020, target_year=2050):
        self.base_year = base_year
        self.target_year = target_year
        self.sbti_pathways = {
            '1.5C': {
                'reduction_rate': 0.074,  # 7.4% annual reduction for 1.5°C
                'description': '1.5°C aligned pathway'
            },
            '2C': {
                'reduction_rate': 0.025,  # 2.5% annual reduction for 2°C
                'description': '2°C aligned pathway'
            }
        }
        
        # Sector-specific reduction rates (example values)
        self.sector_pathways = {
            'Commercial': {
                '1.5C_rate': 0.08,
                '2C_rate': 0.03,
                'net_zero_year': 2040
            },
            'Residential': {
                '1.5C_rate': 0.07,
                '2C_rate': 0.025,
                'net_zero_year': 2045
            },
            'Industrial': {
                '1.5C_rate': 0.06,
                '2C_rate': 0.02,
                'net_zero_year': 2050
            },
            'Transportation': {
                '1.5C_rate': 0.09,
                '2C_rate': 0.035,
                'net_zero_year': 2035
            }
        }
    
    def calculate_pathway(self, base_emissions, pathway_type='1.5C', sector=None, custom_rate=None):
        """
        Calculate emissions pathway based on SBTi methodology or custom rate
        
        Parameters:
        - base_emissions: Base year emissions (tCO2e)
        - pathway_type: '1.5C', '2C', or 'custom'
        - sector: Sector name for sector-specific rates
        - custom_rate: Custom annual reduction rate (0-1)
        
        Returns:
        - DataFrame with year-by-year emissions pathway
        """
        years = range(self.base_year, self.target_year + 1)
        pathway = []
        
        # Get reduction rate
        if custom_rate is not None:
            reduction_rate = custom_rate
            pathway_type = 'custom'
        elif sector and sector in self.sector_pathways:
            reduction_rate = self.sector_pathways[sector][f'{pathway_type}_rate']
        else:
            reduction_rate = self.sbti_pathways[pathway_type]['reduction_rate']
        
        current_emissions = base_emissions
        
        for year in years:
            pathway.append({
                'Year': year,
                'Emissions': current_emissions,
                'Reduction_Rate': reduction_rate,
                'Pathway_Type': pathway_type,
                'Sector': sector if sector else 'General'
            })
            
            # Apply annual reduction
            current_emissions *= (1 - reduction_rate)
        
        return pd.DataFrame(pathway)
    
    def calculate_net_zero_year(self, base_emissions, reduction_rate, threshold_percent=1.0):
        """
        Calculate when net zero would be achieved given a reduction rate
        
        Parameters:
        - base_emissions: Base year emissions (tCO2e)
        - reduction_rate: Annual reduction rate (0-1)
        - threshold_percent: Percentage of base emissions to consider as 'net zero'
        
        Returns:
        - Net zero year
        """
        threshold = base_emissions * (threshold_percent / 100)
        current_emissions = base_emissions
        year = self.base_year
        
        while current_emissions > threshold and year <= 2100:
            current_emissions *= (1 - reduction_rate)
            year += 1
        
        return year if current_emissions <= threshold else None
    
    def calculate_required_reduction_rate(self, base_emissions, target_year, threshold_percent=1.0):
        """
        Calculate required annual reduction rate to reach net zero by target year
        
        Parameters:
        - base_emissions: Base year emissions (tCO2e)
        - target_year: Target year for net zero
        - threshold_percent: Percentage of base emissions to consider as 'net zero'
        
        Returns:
        - Required annual reduction rate (0-1)
        """
        if target_year <= self.base_year:
            return 1.0  # Immediate reduction required
        
        threshold = base_emissions * (threshold_percent / 100)
        years = target_year - self.base_year
        
        # Solve: threshold = base_emissions * (1 - rate)^years
        # rate = 1 - (threshold/base_emissions)^(1/years)
        rate = 1 - (threshold / base_emissions) ** (1 / years)
        
        return max(0, min(1, rate))  # Ensure rate is between 0 and 1
    
    def calculate_carbon_budget(self, base_emissions, pathway_type='1.5C'):
        """
        Calculate remaining carbon budget
        
        Parameters:
        - base_emissions: Base year emissions (tCO2e)
        - pathway_type: '1.5C' or '2C'
        
        Returns:
        - Dictionary with budget information
        """
        pathway = self.calculate_pathway(base_emissions, pathway_type)
        
        # Calculate cumulative emissions
        cumulative_emissions = pathway['Emissions'].sum()
        
        # Global carbon budgets (GtCO2e) - simplified
        global_budgets = {
            '1.5C': 500,  # GtCO2e remaining from 2020
            '2C': 1350    # GtCO2e remaining from 2020
        }
        
        # Convert to tonnes and calculate share
        global_budget_tonnes = global_budgets[pathway_type] * 1e9
        
        return {
            'cumulative_emissions': cumulative_emissions,
            'global_budget': global_budget_tonnes,
            'budget_utilization': (cumulative_emissions / global_budget_tonnes) * 100,
            'pathway_type': pathway_type
        }
    
    def calculate_net_zero_timeline(self, base_emissions, sector=None):
        """
        Calculate when net zero would be achieved under different scenarios
        
        Parameters:
        - base_emissions: Base year emissions (tCO2e)
        - sector: Sector name for sector-specific analysis
        
        Returns:
        - Dictionary with net zero years for different pathways
        """
        results = {}
        
        for pathway_type in ['1.5C', '2C']:
            pathway = self.calculate_pathway(base_emissions, pathway_type, sector)
            
            # Find year when emissions reach near zero (< 1% of base)
            threshold = base_emissions * 0.01
            net_zero_data = pathway[pathway['Emissions'] <= threshold]
            
            if len(net_zero_data) > 0:
                net_zero_year = net_zero_data['Year'].iloc[0]
            else:
                # If no year reaches threshold, use the last year
                net_zero_year = pathway['Year'].iloc[-1]
            
            results[pathway_type] = {
                'net_zero_year': net_zero_year,
                'final_emissions': pathway['Emissions'].iloc[-1],
                'reduction_percentage': ((base_emissions - pathway['Emissions'].iloc[-1]) / base_emissions) * 100
            }
        
        return results
    
    def analyze_sector_breakdown(self, emissions_data):
        """
        Analyze emissions by sector and calculate sector-specific pathways
        
        Parameters:
        - emissions_data: DataFrame with emissions and sector information
        
        Returns:
        - Dictionary with sector analysis
        """
        sector_analysis = {}
        
        # Use lowercase for emissions column
        emissions_col = 'emissions' if 'emissions' in emissions_data.columns else emissions_data.columns[[c.lower() == 'emissions' for c in emissions_data.columns]][0]
        
        for sector in emissions_data['Sector'].unique():
            sector_emissions = emissions_data[emissions_data['Sector'] == sector][emissions_col].sum()
            
            # Calculate pathways for this sector
            sector_analysis[sector] = {
                'base_emissions': sector_emissions,
                'pathways': {
                    '1.5C': self.calculate_pathway(sector_emissions, '1.5C', sector),
                    '2C': self.calculate_pathway(sector_emissions, '2C', sector)
                },
                'net_zero_timeline': self.calculate_net_zero_timeline(sector_emissions, sector),
                'carbon_budget': self.calculate_carbon_budget(sector_emissions, '1.5C')
            }
        
        return sector_analysis

def load_and_prepare_data():
    """
    Load cleaned emissions data and prepare for net zero analysis
    """
    try:
        # Load cleaned data
        entity_a = pd.read_csv('data/processed/cleaned_EntityA.csv')
        entity_b = pd.read_csv('data/processed/cleaned_EntityB.csv')
        
        print("✅ Data loaded successfully")
        print(f"Entity A: {len(entity_a)} records")
        print(f"Entity B: {len(entity_b)} records")
        
        # Add Date column if not present
        for df, name in [(entity_a, 'Entity A'), (entity_b, 'Entity B')]:
            if 'Date' not in df.columns and 'fiscalyear' in df.columns:
                # Handle fiscal year conversion more robustly
                try:
                    df['Date'] = pd.to_datetime(df['fiscalyear'], format='%Y')
                except:
                    # If fiscal year is not in YYYY format, try to extract year
                    df['Date'] = pd.to_datetime(df['fiscalyear'].astype(str).str[:4], format='%Y')
                print(f"✅ Added Date column to {name}")
        
        return entity_a, entity_b
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def create_sample_data():
    """
    Create sample data for demonstration if real data is not available
    """
    np.random.seed(42)
    
    # Sample data structure
    years = range(2020, 2025)
    sectors = ['Commercial', 'Residential', 'Industrial', 'Transportation']
    
    data = []
    for year in years:
        for sector in sectors:
            base_emissions = np.random.uniform(1000, 10000)
            data.append({
                'Date': pd.to_datetime(f'{year}-01-01'),
                'fiscalyear': year,
                'Sector': sector,
                'Emissions': base_emissions,
                'Property': f'Property_{np.random.randint(1, 10)}',
                'Emissions_Source': f'Source_{np.random.randint(1, 5)}'
            })
    
    return pd.DataFrame(data)

def main_analysis():
    """
    Main analysis function
    """
    print("🚀 Starting Net Zero Pathway Analysis")
    print("=" * 50)
    
    # Load data
    entity_a, entity_b = load_and_prepare_data()
    
    if entity_a is None or entity_b is None:
        print("⚠️ Using sample data for demonstration")
        entity_a = create_sample_data()
        entity_b = create_sample_data()
    
    # Initialize pathway calculator
    pathway_calc = NetZeroPathway(base_year=2020, target_year=2050)
    
    # Analyze each entity
    entities = {'Entity A': entity_a, 'Entity B': entity_b}
    results = {}
    current_trajectories = {}
    
    for entity_name, data in entities.items():
        print(f"\n📊 Analyzing {entity_name}")
        print("-" * 30)
        
        # Determine correct column names
        emissions_col = 'emissions'
        year_col = 'fiscalyear'
        if entity_name == 'Entity A':
            sector_col = 'sector'
        else:
            # Prefer 'property_type' if present, else fallback to 'primary_category'
            sector_col = 'property_type' if 'property_type' in data.columns else 'primary_category'
        
        # Calculate total emissions by year
        yearly_emissions = data.groupby(year_col)[emissions_col].sum().reset_index()
        base_emissions = yearly_emissions[emissions_col].mean()  # Use average as base
        
        print(f"Base emissions (avg): {base_emissions:,.0f} tCO2e")
        
        # Calculate current trajectory
        current_trajectory = calculate_current_trajectory(data, entity_name)
        current_trajectories[entity_name] = current_trajectory
        print(f"Current trend: {current_trajectory['trend_description']}")
        print(f"Current net zero year: {current_trajectory['net_zero_year']}")
        
        # Calculate pathways
        pathway_1_5c = pathway_calc.calculate_pathway(base_emissions, '1.5C')
        pathway_2c = pathway_calc.calculate_pathway(base_emissions, '2C')
        
        # Calculate net zero timeline
        net_zero_timeline = pathway_calc.calculate_net_zero_timeline(base_emissions)
        
        # Calculate carbon budget
        carbon_budget = pathway_calc.calculate_carbon_budget(base_emissions, '1.5C')
        
        # For sector analysis, rename sector column to 'Sector' for compatibility
        data_sector = data.rename(columns={sector_col: 'Sector'})
        sector_analysis = pathway_calc.analyze_sector_breakdown(data_sector)
        
        results[entity_name] = {
            'base_emissions': base_emissions,
            'pathway_1_5c': pathway_1_5c,
            'pathway_2c': pathway_2c,
            'net_zero_timeline': net_zero_timeline,
            'carbon_budget': carbon_budget,
            'sector_analysis': sector_analysis
        }
        
        # Print summary
        print(f"1.5°C Net Zero Year: {net_zero_timeline['1.5C']['net_zero_year']}")
        print(f"2°C Net Zero Year: {net_zero_timeline['2C']['net_zero_year']}")
        print(f"Carbon Budget Utilization: {carbon_budget['budget_utilization']:.2f}%")
    
    return results, current_trajectories

def create_visualizations(results, current_trajectories):
    """
    Create comprehensive visualizations for net zero pathways with current trajectory overlay
    """
    print("\n📈 Creating Visualizations")
    print("-" * 30)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Net Zero Pathway Analysis with Current Trajectory', fontsize=16, fontweight='bold')
    
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (entity_name, result) in enumerate(results.items()):
        # Pathway comparison with current trajectory
        ax1 = axes[0, 0] if i == 0 else axes[0, 1]
        current_traj = current_trajectories[entity_name]
        
        # Plot current trajectory
        current_df = current_traj['trajectory']
        current_df = current_df[(current_df['Year'] >= 2020) & (current_df['Year'] <= 2050)]
        ax1.plot(current_df['Year'], current_df['Emissions'], 
                label=f'{entity_name} - Current Trend', color='#FF8C00', linewidth=3, linestyle='-')
        
        # Plot SBTi pathways
        for pathway, label, style in [
            (result['pathway_1_5c'], f'{entity_name} - 1.5°C', '-'),
            (result['pathway_2c'], f'{entity_name} - 2°C', '--')
        ]:
            df = pathway.copy()
            df = df[(df['Year'] >= 2020) & (df['Year'] <= 2050)]
            print(f"Plotting {len(df)} points for {label}")
            ax1.plot(df['Year'], df['Emissions'], label=label, color=colors[i], linestyle=style, linewidth=2)
        
        ax1.set_xlim(2020, 2050)
        ax1.set_title(f'Emissions Pathway - {entity_name}')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Emissions (tCO2e)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Gap analysis visualization
    ax3 = axes[1, 0]
    entities = list(results.keys())
    
    # Calculate gaps
    gaps_1_5c = []
    gaps_2c = []
    current_years = []
    
    for entity_name in entities:
        current_traj = current_trajectories[entity_name]
        result = results[entity_name]
        
        current_year = current_traj['net_zero_year']
        sbti_1_5c_year = result['net_zero_timeline']['1.5C']['net_zero_year']
        sbti_2c_year = result['net_zero_timeline']['2C']['net_zero_year']
        
        gap_1_5c = current_year - sbti_1_5c_year if current_year and sbti_1_5c_year else 0
        gap_2c = current_year - sbti_2c_year if current_year and sbti_2c_year else 0
        
        gaps_1_5c.append(gap_1_5c)
        gaps_2c.append(gap_2c)
        current_years.append(current_year if current_year else 2100)
    
    x = np.arange(len(entities))
    width = 0.35
    
    ax3.bar(x - width/2, gaps_1_5c, width, label='Gap vs 1.5°C', color='#2E8B57')
    ax3.bar(x + width/2, gaps_2c, width, label='Gap vs 2°C', color='#FF6B6B')
    
    ax3.set_title('Net Zero Year Gap Analysis')
    ax3.set_xlabel('Entity')
    ax3.set_ylabel('Years Behind SBTi Target')
    ax3.set_xticks(x)
    ax3.set_xticklabels(entities)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Reduction needed in 2030
    ax4 = axes[1, 1]
    reduction_1_5c = []
    reduction_2c = []
    
    for entity_name in entities:
        current_traj = current_trajectories[entity_name]
        result = results[entity_name]
        
        current_2030 = current_traj['trajectory'][current_traj['trajectory']['Year'] == 2030]['Emissions'].iloc[0]
        sbti_1_5c_2030 = result['pathway_1_5c'][result['pathway_1_5c']['Year'] == 2030]['Emissions'].iloc[0]
        sbti_2c_2030 = result['pathway_2c'][result['pathway_2c']['Year'] == 2030]['Emissions'].iloc[0]
        
        reduction_1_5c.append(((current_2030 - sbti_1_5c_2030) / current_2030) * 100)
        reduction_2c.append(((current_2030 - sbti_2c_2030) / current_2030) * 100)
    
    bars1 = ax4.bar(x - width/2, reduction_1_5c, width, label='Reduction vs 1.5°C', color='#2E8B57')
    bars2 = ax4.bar(x + width/2, reduction_2c, width, label='Reduction vs 2°C', color='#FF6B6B')
    
    ax4.set_title('Additional Reduction Needed by 2030')
    ax4.set_xlabel('Entity')
    ax4.set_ylabel('Additional Reduction (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(entities)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, reduction_1_5c):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    for bar, value in zip(bars2, reduction_2c):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Try to save with lower DPI first
    try:
        plt.savefig('reports/net_zero_pathway_analysis.png', dpi=150, bbox_inches='tight')
        print("✅ Main visualization saved successfully")
    except ValueError as e:
        print(f"⚠️ Failed to save with DPI=150: {e}")
        try:
            plt.savefig('reports/net_zero_pathway_analysis.png', dpi=72, bbox_inches='tight')
            print("✅ Main visualization saved with lower DPI")
        except Exception as e2:
            print(f"❌ Failed to save visualization: {e2}")
            print("Skipping main visualization save...")
    
    plt.show()
    
    # Sector-specific pathways
    if len(results) > 0:
        # Create sector pathways for all entities
        for entity_name, result in results.items():
            sector_analysis = result['sector_analysis']
            
            # Only plot top 4 sectors by base emissions
            sorted_sectors = sorted(sector_analysis.items(), key=lambda x: x[1]['base_emissions'], reverse=True)
            top_sectors = sorted_sectors[:4]
            if len(sector_analysis) > 4:
                print(f"⚠️ More than 4 sectors found for {entity_name}. Only plotting top 4 by emissions.")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Sector-Specific Pathways - {entity_name}', fontsize=16, fontweight='bold')
            
            for i, (sector, analysis) in enumerate(top_sectors):
                ax = axes[i//2, i%2]
                for pathway, label, style, color in [
                    (analysis['pathways']['1.5C'], '1.5°C', '-', '#2E8B57'),
                    (analysis['pathways']['2C'], '2°C', '--', '#FF6B6B')
                ]:
                    df = pathway.copy()
                    df = df[(df['Year'] >= 2020) & (df['Year'] <= 2050)]
                    ax.plot(df['Year'], df['Emissions'], label=label, color=color, linestyle=style, linewidth=2)
                ax.set_xlim(2020, 2050)
                ax.set_title(f'{sector} Sector')
                ax.set_xlabel('Year')
                ax.set_ylabel('Emissions (tCO2e)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            try:
                plt.savefig(f'reports/sector_pathways_{entity_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
                print(f"✅ Sector visualization saved for {entity_name}")
            except Exception as e:
                print(f"❌ Failed to save sector visualization for {entity_name}: {e}")
            
            plt.show()
        
        # Create combined sector comparison
        print("\n📊 Creating Combined Sector Comparison...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sector Comparison Across Entities', fontsize=16, fontweight='bold')
        
        # Get all unique sectors from both entities
        all_sectors = set()
        for entity_name, result in results.items():
            all_sectors.update(result['sector_analysis'].keys())
        
        # Plot top 4 sectors across both entities
        sector_totals = {}
        for entity_name, result in results.items():
            for sector, analysis in result['sector_analysis'].items():
                if sector not in sector_totals:
                    sector_totals[sector] = 0
                sector_totals[sector] += analysis['base_emissions']
        
        top_sectors_combined = sorted(sector_totals.items(), key=lambda x: x[1], reverse=True)[:4]
        
        for i, (sector, total_emissions) in enumerate(top_sectors_combined):
            ax = axes[i//2, i%2]
            
            for entity_name, result in results.items():
                if sector in result['sector_analysis']:
                    analysis = result['sector_analysis'][sector]
                    pathway_1_5c = analysis['pathways']['1.5C']
                    pathway_2c = analysis['pathways']['2C']
                    
                    df_1_5c = pathway_1_5c[(pathway_1_5c['Year'] >= 2020) & (pathway_1_5c['Year'] <= 2050)]
                    df_2c = pathway_2c[(pathway_2c['Year'] >= 2020) & (pathway_2c['Year'] <= 2050)]
                    
                    ax.plot(df_1_5c['Year'], df_1_5c['Emissions'], 
                           label=f'{entity_name} - 1.5°C', linestyle='-', linewidth=2)
                    ax.plot(df_2c['Year'], df_2c['Emissions'], 
                           label=f'{entity_name} - 2°C', linestyle='--', linewidth=2)
            
            ax.set_xlim(2020, 2050)
            ax.set_title(f'{sector} Sector - Combined')
            ax.set_xlabel('Year')
            ax.set_ylabel('Emissions (tCO2e)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.savefig('reports/sector_comparison_combined.png', dpi=150, bbox_inches='tight')
            print("✅ Combined sector comparison saved")
        except Exception as e:
            print(f"❌ Failed to save combined sector comparison: {e}")
        
        plt.show()

def generate_tables(results, current_trajectories):
    """
    Generate and save analysis tables
    """
    print("\n📊 Generating Analysis Tables")
    print("-" * 30)
    
    # Create gap analysis table
    gap_table = create_gap_analysis_table(results, current_trajectories)
    gap_table.to_csv('reports/gap_analysis_table.csv', index=False)
    print("✅ Gap analysis table saved to reports/gap_analysis_table.csv")
    
    # Create pathway comparison table
    pathway_table = create_pathway_comparison_table(results, current_trajectories)
    pathway_table.to_csv('reports/pathway_comparison_table.csv', index=False)
    print("✅ Pathway comparison table saved to reports/pathway_comparison_table.csv")
    
    # Display summary tables
    print("\n📋 Gap Analysis Summary:")
    print(gap_table[['Entity', 'Current_Net_Zero_Year', 'SBTi_1.5C_Net_Zero_Year', 'Gap_vs_1.5C_Years', 'Reduction_Needed_2030_1.5C']].to_string(index=False))
    
    return gap_table, pathway_table

def create_gap_analysis_table(results, current_trajectories):
    """
    Create a comprehensive table showing gap analysis between current trajectory and SBTi pathways
    
    Parameters:
    - results: Dictionary with SBTi pathway results
    - current_trajectories: Dictionary with current trajectory results
    
    Returns:
    - DataFrame with gap analysis
    """
    gap_data = []
    
    for entity_name, result in results.items():
        current_traj = current_trajectories[entity_name]
        
        # Get SBTi net zero years
        sbti_1_5c_year = result['net_zero_timeline']['1.5C']['net_zero_year']
        sbti_2c_year = result['net_zero_timeline']['2C']['net_zero_year']
        current_year = current_traj['net_zero_year']
        
        # Calculate gaps (negative means current trajectory is ahead of SBTi)
        gap_1_5c = current_year - sbti_1_5c_year if current_year and sbti_1_5c_year else None
        gap_2c = current_year - sbti_2c_year if current_year and sbti_2c_year else None
        
        # Get emissions in 2030 and 2050 for different pathways
        current_2030 = current_traj['trajectory'][current_traj['trajectory']['Year'] == 2030]['Emissions'].iloc[0]
        current_2050 = current_traj['trajectory'][current_traj['trajectory']['Year'] == 2050]['Emissions'].iloc[0]
        
        sbti_1_5c_2030 = result['pathway_1_5c'][result['pathway_1_5c']['Year'] == 2030]['Emissions'].iloc[0]
        sbti_1_5c_2050 = result['pathway_1_5c'][result['pathway_1_5c']['Year'] == 2050]['Emissions'].iloc[0]
        
        sbti_2c_2030 = result['pathway_2c'][result['pathway_2c']['Year'] == 2030]['Emissions'].iloc[0]
        sbti_2c_2050 = result['pathway_2c'][result['pathway_2c']['Year'] == 2050]['Emissions'].iloc[0]
        
        # Calculate reduction needed (handle cases where current emissions are already below SBTi)
        reduction_1_5c = ((current_2030 - sbti_1_5c_2030) / current_2030) * 100 if current_2030 > 0 else 0
        reduction_2c = ((current_2030 - sbti_2c_2030) / current_2030) * 100 if current_2030 > 0 else 0
        
        gap_data.append({
            'Entity': entity_name,
            'Base_Emissions_2020': result['base_emissions'],
            'Current_Trend': current_traj['trend_description'],
            'Current_Net_Zero_Year': current_year,
            'SBTi_1.5C_Net_Zero_Year': sbti_1_5c_year,
            'SBTi_2C_Net_Zero_Year': sbti_2c_year,
            'Gap_vs_1.5C_Years': gap_1_5c,
            'Gap_vs_2C_Years': gap_2c,
            'Current_Emissions_2030': current_2030,
            'SBTi_1.5C_Emissions_2030': sbti_1_5c_2030,
            'SBTi_2C_Emissions_2030': sbti_2c_2030,
            'Current_Emissions_2050': current_2050,
            'SBTi_1.5C_Emissions_2050': sbti_1_5c_2050,
            'SBTi_2C_Emissions_2050': sbti_2c_2050,
            'Reduction_Needed_2030_1.5C': reduction_1_5c,
            'Reduction_Needed_2030_2C': reduction_2c
        })
    
    return pd.DataFrame(gap_data)

def create_pathway_comparison_table(results, current_trajectories):
    """
    Create a detailed table comparing all pathways year by year
    
    Parameters:
    - results: Dictionary with SBTi pathway results
    - current_trajectories: Dictionary with current trajectory results
    
    Returns:
    - DataFrame with pathway comparison
    """
    comparison_data = []
    
    for entity_name, result in results.items():
        current_traj = current_traj = current_trajectories[entity_name]
        
        # Merge all pathways
        current_df = current_traj['trajectory'].copy()
        current_df['Pathway'] = 'Current_Trend'
        current_df['Entity'] = entity_name
        
        sbti_1_5c_df = result['pathway_1_5c'].copy()
        sbti_1_5c_df['Pathway'] = 'SBTi_1.5C'
        sbti_1_5c_df['Entity'] = entity_name
        
        sbti_2c_df = result['pathway_2c'].copy()
        sbti_2c_df['Pathway'] = 'SBTi_2C'
        sbti_2c_df['Entity'] = entity_name
        
        # Combine all pathways
        combined_df = pd.concat([current_df, sbti_1_5c_df, sbti_2c_df], ignore_index=True)
        
        # Select key years
        key_years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
        for year in key_years:
            year_data = combined_df[combined_df['Year'] == year]
            for _, row in year_data.iterrows():
                comparison_data.append({
                    'Entity': row['Entity'],
                    'Year': row['Year'],
                    'Pathway': row['Pathway'],
                    'Emissions_tCO2e': row['Emissions']
                })
    
    return pd.DataFrame(comparison_data)

def calculate_current_trajectory(data, entity_name):
    """
    Calculate current trajectory based on historical emissions data
    
    Parameters:
    - data: DataFrame with emissions data
    - entity_name: Name of the entity
    
    Returns:
    - Dictionary with trajectory information
    """
    # Determine correct column names
    emissions_col = 'emissions'
    year_col = 'fiscalyear'
    
    # Calculate yearly emissions
    yearly_emissions = data.groupby(year_col)[emissions_col].sum().reset_index()
    
    # Convert fiscalyear to year if it's datetime
    if yearly_emissions[year_col].dtype == 'datetime64[ns]':
        yearly_emissions['Year'] = yearly_emissions[year_col].dt.year
    else:
        # Try to convert to numeric year
        try:
            yearly_emissions['Year'] = pd.to_numeric(yearly_emissions[year_col])
        except:
            # If conversion fails, try to extract year from string
            yearly_emissions['Year'] = pd.to_numeric(yearly_emissions[year_col].astype(str).str[:4])
    
    # Calculate trend (simple linear regression)
    if len(yearly_emissions) > 1:
        x = yearly_emissions['Year'].values.astype(float)
        y = yearly_emissions[emissions_col].values.astype(float)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Project to 2050
        future_years = range(2020, 2051)
        projected_emissions = [slope * year + intercept for year in future_years]
        
        # Ensure emissions don't go negative
        projected_emissions = [max(0, em) for em in projected_emissions]
        
        trajectory_df = pd.DataFrame({
            'Year': future_years,
            'Emissions': projected_emissions,
            'Trajectory_Type': 'Current_Trend'
        })
        
        # Calculate net zero year for current trajectory
        threshold = yearly_emissions[emissions_col].mean() * 0.01
        net_zero_year = None
        for year, emissions in zip(future_years, projected_emissions):
            if emissions <= threshold:
                net_zero_year = year
                break
        
        return {
            'trajectory': trajectory_df,
            'slope': slope,
            'intercept': intercept,
            'net_zero_year': net_zero_year,
            'historical_data': yearly_emissions,
            'trend_description': f"Annual change: {slope:+.1f} tCO2e/year"
        }
    else:
        # If insufficient data, use flat trajectory
        future_years = range(2020, 2051)
        avg_emissions = yearly_emissions[emissions_col].iloc[0]
        projected_emissions = [avg_emissions] * len(future_years)
        
        trajectory_df = pd.DataFrame({
            'Year': future_years,
            'Emissions': projected_emissions,
            'Trajectory_Type': 'Current_Trend'
        })
        
        return {
            'trajectory': trajectory_df,
            'slope': 0,
            'intercept': avg_emissions,
            'net_zero_year': None,
            'historical_data': yearly_emissions,
            'trend_description': "No trend (insufficient data)"
        }

def interactive_pathway_analysis(base_emissions, reduction_rate=None, target_year=None):
    """
    Interactive pathway analysis with user-defined parameters
    
    Parameters:
    - base_emissions: Base year emissions (tCO2e)
    - reduction_rate: Annual reduction rate (0-1) - if provided, calculate net zero year
    - target_year: Target year for net zero - if provided, calculate required reduction rate
    
    Returns:
    - Dictionary with analysis results
    """
    pathway_calc = NetZeroPathway()
    
    if reduction_rate is not None:
        # Calculate net zero year given reduction rate
        net_zero_year = pathway_calc.calculate_net_zero_year(base_emissions, reduction_rate)
        pathway = pathway_calc.calculate_pathway(base_emissions, 'custom', custom_rate=reduction_rate)
        
        return {
            'type': 'rate_to_year',
            'reduction_rate': reduction_rate,
            'net_zero_year': net_zero_year,
            'pathway': pathway,
            'base_emissions': base_emissions
        }
    
    elif target_year is not None:
        # Calculate required reduction rate given target year
        required_rate = pathway_calc.calculate_required_reduction_rate(base_emissions, target_year)
        pathway = pathway_calc.calculate_pathway(base_emissions, 'custom', custom_rate=required_rate)
        
        return {
            'type': 'year_to_rate',
            'target_year': target_year,
            'required_rate': required_rate,
            'pathway': pathway,
            'base_emissions': base_emissions
        }
    
    else:
        # Default SBTi analysis
        return {
            'type': 'sbti_analysis',
            'pathway_1_5c': pathway_calc.calculate_pathway(base_emissions, '1.5C'),
            'pathway_2c': pathway_calc.calculate_pathway(base_emissions, '2C'),
            'net_zero_timeline': pathway_calc.calculate_net_zero_timeline(base_emissions),
            'carbon_budget': pathway_calc.calculate_carbon_budget(base_emissions, '1.5C'),
            'base_emissions': base_emissions
        }

def generate_report(results):
    """
    Generate comprehensive report
    """
    print("\n📋 Generating Report")
    print("-" * 30)
    
    report = []
    report.append("# Net Zero Pathway Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    for entity_name, result in results.items():
        report.append(f"## {entity_name}")
        report.append("")
        
        # Summary statistics
        report.append("### Summary Statistics")
        report.append(f"- **Base Emissions**: {result['base_emissions']:,.0f} tCO2e")
        report.append(f"- **1.5°C Net Zero Year**: {result['net_zero_timeline']['1.5C']['net_zero_year']}")
        report.append(f"- **2°C Net Zero Year**: {result['net_zero_timeline']['2C']['net_zero_year']}")
        report.append(f"- **Carbon Budget Utilization**: {result['carbon_budget']['budget_utilization']:.2f}%")
        report.append("")
        
        # Sector breakdown
        report.append("### Sector Analysis")
        for sector, analysis in result['sector_analysis'].items():
            report.append(f"#### {sector}")
            report.append(f"- Base Emissions: {analysis['base_emissions']:,.0f} tCO2e")
            report.append(f"- 1.5°C Net Zero: {analysis['net_zero_timeline']['1.5C']['net_zero_year']}")
            report.append(f"- 2°C Net Zero: {analysis['net_zero_timeline']['2C']['net_zero_year']}")
            report.append("")
        
        report.append("---")
        report.append("")
    
    # Add interactive features section
    report.append("## Interactive Features")
    report.append("")
    report.append("### Net Zero Year Calculator")
    report.append("Given an annual reduction rate, calculate when net zero would be achieved.")
    report.append("")
    report.append("### Required Reduction Rate Calculator")
    report.append("Given a target year, calculate the required annual reduction rate.")
    report.append("")
    report.append("### SBTi Comparison")
    report.append("Compare your pathway with science-based targets (1.5°C and 2°C).")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open('reports/net_zero_pathway_report.md', 'w') as f:
        f.write(report_text)
    
    print("✅ Report saved to reports/net_zero_pathway_report.md")
    
    return report_text

if __name__ == "__main__":
    # Run main analysis
    results, current_trajectories = main_analysis()
    
    # Create visualizations
    create_visualizations(results, current_trajectories)
    
    # Generate tables
    gap_table, pathway_table = generate_tables(results, current_trajectories)
    
    # Generate report
    report = generate_report(results)
    
    print("\n🎉 Net Zero Pathway Analysis Complete!")
    print("=" * 50)
    print("📁 Files generated:")
    print("- reports/net_zero_pathway_analysis.png")
    for entity_name in results.keys():
        print(f"- reports/sector_pathways_{entity_name.replace(' ', '_')}.png")
    print("- reports/sector_comparison_combined.png")
    print("- reports/gap_analysis_table.csv")
    print("- reports/pathway_comparison_table.csv")
    print("- reports/net_zero_pathway_report.md")
    
    # Demonstrate interactive features
    print("\n🔧 Interactive Features Demonstration")
    print("=" * 50)
    
    # Example with Entity A base emissions
    example_emissions = 349319
    
    print(f"\n📊 Example Analysis for {example_emissions:,.0f} tCO2e/year")
    print("-" * 40)
    
    # 1. Calculate net zero year for different reduction rates
    print("\n1️⃣ Net Zero Year Calculator:")
    reduction_rates = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
    for rate in reduction_rates:
        result = interactive_pathway_analysis(example_emissions, reduction_rate=rate)
        net_zero_year = result['net_zero_year']
        print(f"   {rate*100:.0f}% annual reduction → Net Zero by {net_zero_year}")
    
    # 2. Calculate required reduction rate for different target years
    print("\n2️⃣ Required Reduction Rate Calculator:")
    target_years = [2030, 2040, 2050, 2060]
    for year in target_years:
        result = interactive_pathway_analysis(example_emissions, target_year=year)
        required_rate = result['required_rate']
        print(f"   Net Zero by {year} → {required_rate*100:.1f}% annual reduction required")
    
    # 3. Compare with SBTi pathways
    print("\n3️⃣ SBTi Comparison:")
    sbti_result = interactive_pathway_analysis(example_emissions)
    print(f"   1.5°C pathway: {sbti_result['net_zero_timeline']['1.5C']['net_zero_year']}")
    print(f"   2°C pathway: {sbti_result['net_zero_timeline']['2C']['net_zero_year']}")
    
    print("\n✅ Interactive features ready for dashboard integration!") 