python notebooks/01_data_exploration.py
Script executed on: 2025-06-30 16:10:00
Pandas version: 1.5.3
NumPy version: 1.23.5
Raw data directory: data\raw
Processed data directory: data\processed
Reports directory: reports

============================================================
1. DATA LOADING AND INITIAL INSPECTION
============================================================
Successfully loaded EntityA data: (142585, 7)
Successfully loaded EntityB data: (109265, 8)

==================================================
ENTITY: EntityA
==================================================

Shape: (142585, 7)

Columns:
   1. Sector
   2. Property Name
   3. Scopes
   4. Emissions Source
   5. Class
   6. Emissions
   7. FiscalYear

Data Types:
Sector               object
Property Name        object
Scopes               object
Emissions Source     object
Class                object
Emissions           float64
FiscalYear           object
dtype: object

First 5 rows:
        Sector      Property Name   Scopes       Emissions Source        Class  Emissions FiscalYear
0  Development  DEVRO PARK (3579)  Scope 2  Electricity generated  Electricity   2.880592       FY20
1  Development  DEVRO PARK (3579)  Scope 2  Electricity generated  Electricity   0.232140       FY21
2  Development  DEVRO PARK (3579)  Scope 2  Electricity generated  Electricity   0.364133       FY22
3  Development  DEVRO PARK (3579)  Scope 2  Electricity generated  Electricity   0.141091       FY23
4  Development  DEVRO PARK (3579)  Scope 2  Electricity generated  Electricity   0.063125       FY24

Last 5 rows:
        Sector                 Property Name   Scopes Emissions Source  Class  Emissions FiscalYear
142580  Office  WOODLANDS OFFICE PARK (3790)  Scope 3          Plastic  Waste   0.009430       FY25
142581  Office  WOODLANDS OFFICE PARK (3790)  Scope 3          Plastic  Waste   0.033766       FY19
142582  Office  WOODLANDS OFFICE PARK (3790)  Scope 3          Plastic  Waste   0.011511       FY20
142583  Office  WOODLANDS OFFICE PARK (3790)  Scope 3          Plastic  Waste   0.009156       FY21
142584  Office  WOODLANDS OFFICE PARK (3790)  Scope 3          Plastic  Waste   0.007725       FY22

==================================================
ENTITY: EntityB
==================================================

Shape: (109265, 8)

Columns:
   1. Property Type
   2. Property Name
   3. Scopes
   4. Emissions Source
   5. Primary Category
   6. Class
   7. Emissions
   8. FiscalYear

Data Types:
Property Type        object
Property Name        object
Scopes               object
Emissions Source     object
Primary Category     object
Class                object
Emissions           float64
FiscalYear           object
dtype: object

First 5 rows:
  Property Type         Property Name   Scopes       Emissions Source       Primary Category        Class  Emissions FiscalYear
0    Commercial  228 Pretorius Street  Scope 2  Electricity generated  Purchased Electricity  Electricity    2.78356       FY21
1    Commercial  228 Pretorius Street  Scope 2  Electricity generated  Purchased Electricity  Electricity    2.48248       FY22
2    Commercial  228 Pretorius Street  Scope 2  Electricity generated  Purchased Electricity  Electricity   -0.38582       FY23
3    Commercial  228 Pretorius Street  Scope 2  Electricity generated  Purchased Electricity  Electricity   13.92992       FY24
4    Commercial  228 Pretorius Street  Scope 2  Electricity generated  Purchased Electricity  Electricity    7.35381       FY25

Last 5 rows:
          Property Type  Property Name   Scopes                 Emissions Source              Primary Category        Class  Emissions FiscalYear
109260  Shopping Centre  Woodmead Mart  Scope 3  WTT- overseas electricity (T&D)  13. Downstream leased assets  Electricity   2.358314       FY21
109261  Shopping Centre  Woodmead Mart  Scope 3  WTT- overseas electricity (T&D)  13. Downstream leased assets  Electricity   2.240824       FY22
109262  Shopping Centre  Woodmead Mart  Scope 3  WTT- overseas electricity (T&D)  13. Downstream leased assets  Electricity   2.309586       FY21
109263  Shopping Centre  Woodmead Mart  Scope 3  WTT- overseas electricity (T&D)  13. Downstream leased assets  Electricity   2.029158       FY22
109264  Shopping Centre  Woodmead Mart  Scope 3  WTT- overseas electricity (T&D)  13. Downstream leased assets  Electricity   2.408014       FY23

... [CONTENT CONTINUES THROUGH ALL STAGES: Data Quality Assessment, Temporal Analysis, Sector Analysis, Emissions Analysis, Summary Report, and Recommendations]

Summary saved to: data\processed\data_exploration_summary.csv

Recommendations saved to: data\processed\cleaning_recommendations.txt

Data exploration completed successfully!
All findings have been documented and saved for reproducibility.