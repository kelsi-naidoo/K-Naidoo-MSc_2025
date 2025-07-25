python notebooks/02_data_cleaning.py
Script executed on: 2025-07-01 14:15:00
Pandas version: 1.5.3
NumPy version: 1.23.5
Processed data directory: data\processed
Figures directory: reports\figures

============================================================
1. DATA LOADING AND INITIAL ASSESSMENT
============================================================
INFO:data.data_loader:Configuration loaded successfully
INFO:data.data_loader:DataLoader initialized with 2 entities
Loading raw data...
INFO:data.data_loader:Loading data for all entities...
INFO:data.data_loader:Loading data from data\raw\Emissions_EntityA.xlsx
INFO:data.data_loader:Successfully loaded EntityA data: (142585, 7)
INFO:data.data_loader:Loading data from data\raw\Emissions_EntityB.xlsx
INFO:data.data_loader:Successfully loaded EntityB data: (109265, 8)
INFO:data.data_loader:Loaded data for 2 entities

Data Loading Summary:
Total entities: 2
Loaded entities: 2
Loading errors: 0

EntityA:
  Shape: (142585, 7)
  Columns: 7
  Memory usage: 57232.72 KB

EntityB:
  Shape: (109265, 8)
  Columns: 8
  Memory usage: 53916.76 KB

Validating data structure...

EntityA:
  Temporal columns: ['FiscalYear']
  Emissions columns: ['Emissions Source', 'Emissions']
  Sector columns: ['Sector']

EntityB:
  Temporal columns: ['FiscalYear']
  Emissions columns: ['Emissions Source', 'Emissions']
  Sector columns: ['Primary Category']

============================================================
2. DATA CLEANING PROCESS
============================================================
Using enhanced data cleaning...

Cleaning EntityA...
  Converting FiscalYear format...
  Converted 142585 valid fiscal years
  Processing emissions data...
  Found 3845 negative emissions (corrections)
  Flagged 3845 corrections for separate processing
  Removed 40824 duplicate rows
  Sorted by FiscalYear
  Original shape: (142585, 7)
  Cleaned shape: (101761, 10)
  Rows removed: 40824
  Columns changed: -3

Cleaning EntityB...
  Converting FiscalYear format...
  Converted 109265 valid fiscal years
  Processing emissions data...
  Found 2968 negative emissions (corrections)
  Flagged 2968 corrections for separate processing
  Removed 43238 duplicate rows
  Sorted by FiscalYear
  Original shape: (109265, 8)
  Cleaned shape: (66027, 11)
  Rows removed: 43238
  Columns changed: -3

============================================================
3. SAVING CLEANED DATA
============================================================
Saved cleaned EntityA data to: data\processed\cleaned_EntityA.csv
Saved cleaned EntityB data to: data\processed\cleaned_EntityB.csv

============================================================
4. DATA VALIDATION
============================================================
INFO:data.data_validator:Validation completed for EntityA: PASS
INFO:data.data_validator:Validation completed for EntityB: PASS

============================================================
5. FEATURE ENGINEERING
============================================================
Features created and saved to:
- data\processed\features_EntityA.csv
- data\processed\features_EntityB.csv

============================================================
6. MACHINE LEARNING PREPARATION
============================================================
Target column: emissions_source
EntityA: 101761 samples
EntityB: 66027 samples

============================================================
7. FINAL EXPORT
============================================================
Cleaned, engineered, and ML-ready data saved successfully.