# Detailed Anomaly Analysis Report - EntityB
Generated: 2025-07-08 19:37:52

## Overview
This report provides detailed analysis of where anomalies are coming from in the EntityB dataset.

## Summary Statistics
- **Total Samples**: 66027
- **Contamination Level**: 0.1% (very strict detection)

## Anomaly Detection Results by Model

### Isolation Forest
- **Anomaly Count**: 63 (0.10%)
- **Emissions Statistics for Anomalies**:
  - Mean: 286.09
  - Std: 225.65
  - Min: 0.01
  - Max: 800.52
  - Median: 303.87

### Lof
- **Anomaly Count**: 67 (0.10%)
- **Emissions Statistics for Anomalies**:
  - Mean: 28.14
  - Std: 129.90
  - Min: 0.00
  - Max: 754.25
  - Median: 0.08

### Elliptic Envelope
- **Anomaly Count**: 67 (0.10%)
- **Emissions Statistics for Anomalies**:
  - Mean: 26.73
  - Std: 65.67
  - Min: 0.00
  - Max: 354.40
  - Median: 2.15

## Emission Source Analysis

### Isolation Forest
- **Electricity generated**: 56 anomalies
- **Water supply**: 2 anomalies
- **T&D- overseas electricity**: 1 anomalies
- **WTT- overseas electricity (T&D)**: 1 anomalies
- **Diesel**: 1 anomalies
- **Other**: 1 anomalies
- **Montreal protocol products**: 1 anomalies

### Lof
- **Water supply**: 22 anomalies
- **Electricity generated**: 11 anomalies
- **Plastic**: 11 anomalies
- **Other**: 7 anomalies
- **WTT- overseas electricity (T&D)**: 6 anomalies
- **General Landfill**: 5 anomalies
- **T&D- overseas electricity**: 2 anomalies
- **Diesel**: 2 anomalies
- **WTT- overseas electricity (generation)**: 1 anomalies

### Elliptic Envelope
- **Electricity generated**: 35 anomalies
- **T&D- overseas electricity**: 14 anomalies
- **WTT- overseas electricity (T&D)**: 9 anomalies
- **WTT- overseas electricity (generation)**: 5 anomalies
- **Liquid fuels**: 2 anomalies
- **Diesel**: 1 anomalies
- **General Landfill**: 1 anomalies

## Property Analysis

### Isolation Forest

#### property_type:
- **Commercial**: 31 anomalies
- **Mixed**: 13 anomalies
- **Shopping Centre**: 13 anomalies
- **Industrial**: 6 anomalies

#### property_name:
- **Killarney Mall**: 11 anomalies
- **Sharon's Place**: 9 anomalies
- **Shoprite**: 8 anomalies
- **Louis Pasteur (2) Louis Pasteur Medical Centre**: 7 anomalies
- **Steyns Industrial Park**: 5 anomalies
- **The Tannery Industrial Park**: 3 anomalies
- **Prinschurch**: 3 anomalies
- **The Fields**: 3 anomalies
- **Mr Price**: 2 anomalies
- **Klamson Towers**: 2 anomalies
- **Gezina City**: 2 anomalies
- **Du Proes (5) (Cnr Proes and Du Toit)**: 1 anomalies
- **Metromitch(3) Charlotte Maxeke Street**: 1 anomalies
- **Du Proes (3) (Cnr Du Toit and Shepard)**: 1 anomalies
- **Rezmep (5) (Pretorius Street 2)**: 1 anomalies
- **Ross Electrical**: 1 anomalies
- **Soutwest Properties**: 1 anomalies
- **3 West Street**: 1 anomalies
- **Kempton Place**: 1 anomalies

### Lof

#### property_type:
- **Commercial**: 37 anomalies
- **Mixed**: 11 anomalies
- **Shopping Centre**: 11 anomalies
- **Residential**: 5 anomalies
- **Industrial**: 3 anomalies

#### property_name:
- **Blaauw Village (JV)**: 4 anomalies
- **Motor City Strijdom Park**: 3 anomalies
- **Louis Pasteur (1) Health Connect**: 3 anomalies
- **Nzunza House**: 3 anomalies
- **Arlington House**: 2 anomalies
- **Woodmead Mart**: 2 anomalies
- **Corner Place**: 2 anomalies
- **Prinstruben**: 2 anomalies
- **Gezina City**: 2 anomalies
- **Focus House**: 2 anomalies
- **Vuselela Place**: 2 anomalies
- **Killarney Mall**: 2 anomalies
- **Prinschurch**: 2 anomalies
- **Plaza Place**: 1 anomalies
- **Rentmeester Park**: 1 anomalies
- **Kempton Place**: 1 anomalies
- **Carlzeil (2) (Zeiler Street)**: 1 anomalies
- **Lusam Mansions**: 1 anomalies
- **Toitman**: 1 anomalies
- **Prinsman Place (1)**: 1 anomalies
- **Jeppe House**: 1 anomalies
- **Capitol Towers North**: 1 anomalies
- **Crown Court (Ischurch2)**: 1 anomalies
- **Jardown (1) (Downies)**: 1 anomalies
- **Jardown (2) (Dome)**: 1 anomalies
- **Lasmitch Properties**: 1 anomalies
- **FNB Centurion**: 1 anomalies
- **Kyalami Crescent**: 1 anomalies
- **Klamson Towers**: 1 anomalies
- **Selmar**: 1 anomalies
- **Prime Towers**: 1 anomalies
- **Rovon Investments**: 1 anomalies
- **Henwoods**: 1 anomalies
- **Howzit Hilda**: 1 anomalies
- **Waverley Plaza**: 1 anomalies
- **Silver Place**: 1 anomalies
- **Wits Technikon**: 1 anomalies
- **Tomzeil**: 1 anomalies
- **Panag Investments**: 1 anomalies
- **City Corner (1)**: 1 anomalies
- **Mr Price**: 1 anomalies
- **Pretjolum (4) (Warehouses)**: 1 anomalies
- **Tali's Place**: 1 anomalies
- **The Brooklyn**: 1 anomalies
- **Craig's Place**: 1 anomalies
- **Rapier**: 1 anomalies
- **Rezmep (8) (Pretorius Street 1)**: 1 anomalies
- **BP Leyds Street**: 1 anomalies
- **Station Place**: 1 anomalies

### Elliptic Envelope

#### property_type:
- **Commercial**: 34 anomalies
- **Mixed**: 24 anomalies
- **Shopping Centre**: 6 anomalies
- **Industrial**: 3 anomalies

#### property_name:
- **Sharon's Place**: 5 anomalies
- **Silver Place**: 3 anomalies
- **Cuthchurch (1) (Andries Street)**: 3 anomalies
- **Waverley Plaza**: 3 anomalies
- **Savyon Place**: 2 anomalies
- **Louis Pasteur (2) Louis Pasteur Medical Centre**: 2 anomalies
- **City Corner (3)**: 2 anomalies
- **Provisus**: 2 anomalies
- **Henwoods**: 2 anomalies
- **Station Place**: 2 anomalies
- **City Corner (2)**: 2 anomalies
- **Sildale Park**: 2 anomalies
- **The Fields**: 2 anomalies
- **250 Pretorius Street**: 2 anomalies
- **Brianley (3) (Tomkordale Building)**: 1 anomalies
- **Steynscor**: 1 anomalies
- **Unity Heights**: 1 anomalies
- **Vuselela Place**: 1 anomalies
- **Gezina City**: 1 anomalies
- **Pete's Place**: 1 anomalies
- **The Park Shopping Centre**: 1 anomalies
- **3 West Street**: 1 anomalies
- **Blaauw Village (JV)**: 1 anomalies
- **Central Towers**: 1 anomalies
- **Steyns Place**: 1 anomalies
- **Reliance Centre**: 1 anomalies
- **228 Pretorius Street**: 1 anomalies
- **The Brooklyn**: 1 anomalies
- **Elephant House**: 1 anomalies
- **Splendid Place**: 1 anomalies
- **Inner Court**: 1 anomalies
- **Nedwest Centre**: 1 anomalies
- **Provincial House**: 1 anomalies
- **Marlborough House**: 1 anomalies
- **Lutbridge (2) Cnr Court and WF Nkomo Street**: 1 anomalies
- **Shoprite**: 1 anomalies
- **Scott's Corner**: 1 anomalies
- **Curpro**: 1 anomalies
- **Nedbank Plaza**: 1 anomalies
- **Royal Place**: 1 anomalies
- **Bram Fischer Towers**: 1 anomalies
- **Time Place**: 1 anomalies
- **Apollo Centre**: 1 anomalies
- **Cuthchurch (2) (Chambers)**: 1 anomalies
- **Wits Technikon**: 1 anomalies
- **Union Club**: 1 anomalies
- **Rentmeester Park**: 1 anomalies

## Temporal Analysis

### Isolation Forest

#### By Year:
- **2021**: 9 anomalies
- **2022**: 24 anomalies
- **2023**: 4 anomalies
- **2024**: 14 anomalies
- **2025**: 12 anomalies

#### By Month:
- **Jan**: 63 anomalies

#### By Quarter:
- **Q1**: 63 anomalies

### Lof

#### By Year:
- **2021**: 1 anomalies
- **2022**: 5 anomalies
- **2023**: 10 anomalies
- **2024**: 50 anomalies
- **2025**: 1 anomalies

#### By Month:
- **Jan**: 67 anomalies

#### By Quarter:
- **Q1**: 67 anomalies

### Elliptic Envelope

#### By Year:
- **2021**: 32 anomalies
- **2022**: 9 anomalies
- **2023**: 26 anomalies

#### By Month:
- **Jan**: 67 anomalies

#### By Quarter:
- **Q1**: 67 anomalies

## Consensus Anomalies (Detected by Multiple Models)
- **Count**: 4 (0.01%)
- **Emissions Statistics**:
  - Mean: 67.42
  - Std: 106.61
  - Min: 0.04
  - Max: 226.62
  - Median: 21.52

### Consensus Anomalies by Source:
- **Electricity generated**: 3 anomalies
- **Water supply**: 1 anomalies

### Consensus Anomalies by Time Period:

#### By Year:
- **2022**: 1 anomalies
- **2023**: 2 anomalies
- **2024**: 1 anomalies

## Key Insights and Recommendations

### High-Priority Investigations:
1. **Consensus Anomalies**: Focus on anomalies detected by multiple models as they are most likely to be real issues
2. **Temporal Patterns**: Investigate time periods with high anomaly concentrations
3. **Source-Specific Issues**: Examine emission sources with disproportionate anomaly rates
4. **Property-Specific Issues**: Look into properties with unusual anomaly patterns

### Data Quality Actions:
1. **Verify Extreme Values**: Check if high-emission anomalies represent actual data or measurement errors
2. **Review Negative Emissions**: Investigate negative emission values for potential corrections
3. **Validate Source Data**: Ensure emission source classifications are accurate
4. **Cross-Reference Properties**: Verify property information consistency

### Operational Recommendations:
1. **Monitoring Focus**: Increase monitoring frequency for high-anomaly sources/properties
2. **Process Review**: Investigate operational processes during high-anomaly periods
3. **Equipment Checks**: Review equipment performance during anomaly periods
4. **Staff Training**: Provide additional training for data collection during critical periods
