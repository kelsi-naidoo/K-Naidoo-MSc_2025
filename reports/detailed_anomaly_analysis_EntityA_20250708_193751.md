# Detailed Anomaly Analysis Report - EntityA
Generated: 2025-07-08 19:37:51

## Overview
This report provides detailed analysis of where anomalies are coming from in the EntityA dataset.

## Summary Statistics
- **Total Samples**: 101761
- **Contamination Level**: 0.1% (very strict detection)

## Anomaly Detection Results by Model

### Isolation Forest
- **Anomaly Count**: 93 (0.09%)
- **Emissions Statistics for Anomalies**:
  - Mean: 486.40
  - Std: 529.67
  - Min: 0.00
  - Max: 1732.38
  - Median: 190.53

### Lof
- **Anomaly Count**: 102 (0.10%)
- **Emissions Statistics for Anomalies**:
  - Mean: 0.21
  - Std: 1.75
  - Min: 0.00
  - Max: 17.65
  - Median: 0.00

### Elliptic Envelope
- **Anomaly Count**: 102 (0.10%)
- **Emissions Statistics for Anomalies**:
  - Mean: 38.00
  - Std: 92.18
  - Min: 0.00
  - Max: 590.96
  - Median: 0.60

## Emission Source Analysis

### Isolation Forest
- **Electricity generated**: 67 anomalies
- **Water supply**: 17 anomalies
- **Other**: 3 anomalies
- **Liquid fuels**: 2 anomalies
- **General Landfill**: 1 anomalies
- **Municipal waste**: 1 anomalies
- **Metal**: 1 anomalies
- **Water**: 1 anomalies

### Lof
- **Electricity generated**: 32 anomalies
- **Water supply**: 16 anomalies
- **Paper**: 11 anomalies
- **Metal**: 11 anomalies
- **Other**: 10 anomalies
- **Plastic**: 10 anomalies
- **Municipal waste**: 5 anomalies
- **Liquid fuels**: 5 anomalies
- **Water**: 1 anomalies
- **Refuse**: 1 anomalies

### Elliptic Envelope
- **Electricity generated**: 61 anomalies
- **Water supply**: 17 anomalies
- **General Landfill**: 8 anomalies
- **Paper**: 5 anomalies
- **Metal**: 3 anomalies
- **Plastic**: 3 anomalies
- **Liquid fuels**: 2 anomalies
- **Water**: 1 anomalies
- **Municipal waste**: 1 anomalies
- **Other**: 1 anomalies

## Property Analysis

### Isolation Forest

#### sector:
- **Industrial**: 47 anomalies
- **Office**: 43 anomalies
- **Healthcare**: 2 anomalies
- **Development**: 1 anomalies

#### property_name:
- **WADESTONE INDUSTRIAL PARK (2160)**: 10 anomalies
- **RUNWAY PARK (3124)**: 9 anomalies
- **CONSTANTIA PARK (1813)**: 7 anomalies
- **THE TERRACES (3569)**: 6 anomalies
- **OMNI PARK (3112)**: 5 anomalies
- **OXFORD 144 – GPT (2228)**: 5 anomalies
- **WATERFALL PARK (2473)**: 3 anomalies
- **WATERFALL AUGRABIES (2424)**: 3 anomalies
- **N1 MEDICAL CHAMBERS - HEALTH (1047)**: 2 anomalies
- **THE PARK ON 16TH BLOCKS DEF (2462)**: 2 anomalies
- **SALIGNA (2186)**: 2 anomalies
- **28 SACKS CIRCLE (2196)**: 2 anomalies
- **57 ON GIBSON (1041)**: 2 anomalies
- **ROGGEBAAI PLACE (3788)**: 2 anomalies
- **TYGERBERG P/THOP PHASE 1,2&4 (3751)**: 2 anomalies
- **HILLTOP INDUSTRIAL PARK (3025)**: 2 anomalies
- **WILLOWBRIDGE PLACE (2151)**: 1 anomalies
- **1 FRIESLAND DRIVE (2150)**: 1 anomalies
- **SCIENTIA (2128)**: 1 anomalies
- **PROLECON (3131)**: 1 anomalies
- **THE GROVE BUSINESS ESTATE (2137)**: 1 anomalies
- **RIVONIA CROSSING 1 - GPT (2214)**: 1 anomalies
- **82 GRAYSTON DRIVE (3707)**: 1 anomalies
- **MONTCLARE PLACE (3566)**: 1 anomalies
- **PS PROPS (3138)**: 1 anomalies
- **ROUTE 41 (1843)**: 1 anomalies
- **OUDE MOULEN (3122)**: 1 anomalies
- **1 SIXTY JAN SMUTS (3545)**: 1 anomalies
- **10 ST ANDREWS ROAD (1912)**: 1 anomalies
- **WESTMEAD INDUSTRIAL PARK (3097)**: 1 anomalies
- **HEALTHCARE PARK (2437)**: 1 anomalies
- **STORMAIN (3012)**: 1 anomalies
- **WOODLANDS OFFICE PARK (3790)**: 1 anomalies
- **RUNWAY PARK CO OWNERSHIP (7033)**: 1 anomalies
- **INYANDA1,3 & 4  - 100% (2922)**: 1 anomalies
- **STORMILL 51 (4352)**: 1 anomalies
- **BOND TOWER (2479)**: 1 anomalies
- **PROTRANS (3089)**: 1 anomalies
- **GOLD REEF PARK (3110)**: 1 anomalies
- **ALBION SPRINGS (3708)**: 1 anomalies
- **RIDGEVIEW UMHLANGA (2159)**: 1 anomalies
- **100 WEST STREET (2164)**: 1 anomalies
- **DEVRO PARK (3579)**: 1 anomalies
- **23 HERMAN ROAD (2442)**: 1 anomalies
- **GIE 1 PORTIONS 1 & 2 OF ERF308 (2484)**: 1 anomalies

### Lof

#### sector:
- **Office**: 61 anomalies
- **Industrial**: 41 anomalies

#### property_name:
- **HAMMARSDALE (2224)**: 7 anomalies
- **OXFORD 144 – GPT (2228)**: 4 anomalies
- **HOMESTEAD PARK (3529)**: 4 anomalies
- **151 ON 5TH (2113)**: 3 anomalies
- **CONSTANTIA PARK (1813)**: 3 anomalies
- **PARAMOUNT PLACE (3559)**: 3 anomalies
- **1 SIXTY JAN SMUTS (3545)**: 3 anomalies
- **ROSEBANK OFFICE PARK (2406)**: 2 anomalies
- **AUTUMN ROAD (3540)**: 2 anomalies
- **24 FLANDERS DRIVE (3592)**: 2 anomalies
- **RIDGEVIEW UMHLANGA (2159)**: 2 anomalies
- **GROSVENOR CORNER (1823)**: 2 anomalies
- **GILLOOLYS VIEW (1819)**: 2 anomalies
- **SUNNYSIDE OFFICE PARK (1846)**: 2 anomalies
- **WINGFIELD (3141)**: 1 anomalies
- **MIDRAND CENTRAL BUS PARK 520 (2180)**: 1 anomalies
- **31 IMPALA ROAD (2190)**: 1 anomalies
- **GROWTHPOINT BUSINESS PARK (1934)**: 1 anomalies
- **44 ON GRAND CENTRAL (2471)**: 1 anomalies
- **MIDWAY PLACE (2440)**: 1 anomalies
- **GRENVILLE (3125)**: 1 anomalies
- **INDEPENDENCE SQUARE (3554)**: 1 anomalies
- **3012A WMM (3730)**: 1 anomalies
- **4 PENCARROW - GPT (2208)**: 1 anomalies
- **MIDRAND CENTRAL BUS PARK 519 (2179)**: 1 anomalies
- **200 ON MAIN (2415)**: 1 anomalies
- **WADESTONE INDUSTRIAL PARK (2160)**: 1 anomalies
- **LOPER CORNER (3169)**: 1 anomalies
- **35 IMPALA ROAD (2187)**: 1 anomalies
- **11B RILEY ROAD (2427)**: 1 anomalies
- **CENTRALPOINT POA OFFICE (2192)**: 1 anomalies
- **LUMLEY HOUSE (2410)**: 1 anomalies
- **COROBRICK (2402)**: 1 anomalies
- **MENLYN PIAZZA (2432)**: 1 anomalies
- **BOFORS 2 (3036)**: 1 anomalies
- **ALTERNATOR (3040)**: 1 anomalies
- **GALLAGHER PLACE (1919)**: 1 anomalies
- **CENTRAL PARK - MIDRAND (1811)**: 1 anomalies
- **GIE 1 PORTIONS 1 & 2 OF ERF308 (2484)**: 1 anomalies
- **ILLOVO BOULEVARD PIAZZAS (3720)**: 1 anomalies
- **HILLTOP INDUSTRIAL PARK (3025)**: 1 anomalies
- **ISANDO 103 (4361)**: 1 anomalies
- **EASTGATE BUSINESS PARK (2405)**: 1 anomalies
- **THE DISTRICT (1894)**: 1 anomalies
- **NAUTICA (3725)**: 1 anomalies
- **DITSELA PLACE (1889)**: 1 anomalies
- **7 WESSELS ROAD (2914)**: 1 anomalies
- **EPPING 4 (2123)**: 1 anomalies
- **ENGINE AVENUE (3046)**: 1 anomalies
- **FERNTOWERS (4358)**: 1 anomalies
- **ADT HOUSE (2414)**: 1 anomalies
- **STORMAIN (3012)**: 1 anomalies
- **GOLD REEF PARK (3110)**: 1 anomalies
- **ROUTE 24 (1842)**: 1 anomalies
- **10 ST ANDREWS ROAD (1912)**: 1 anomalies
- **HEWETT (3034)**: 1 anomalies
- **ALBERT AMON 212 (3176)**: 1 anomalies
- **HOMESTEAD PLACE (1924)**: 1 anomalies
- **INYANDA 2 (2923)**: 1 anomalies
- **CITY DEEP INDUSTRIAL PARK (2408)**: 1 anomalies
- **ST DAVID'S PARK (2111)**: 1 anomalies
- **MORNINGSIDE 1331 (2165)**: 1 anomalies
- **NEWLANDS ON MAIN (2422)**: 1 anomalies
- **TRANSFIELD (3185)**: 1 anomalies
- **THE TERRACES (3569)**: 1 anomalies
- **GATEWAY (3061)**: 1 anomalies
- **EPPING 6 (2125)**: 1 anomalies
- **GEMINI (4367)**: 1 anomalies
- **OGILVY BUILDING (2468)**: 1 anomalies
- **THE BOULEVARD UMHLANGA (1895)**: 1 anomalies
- **MOUNT JOY (3027)**: 1 anomalies
- **PETER PLACE (1933)**: 1 anomalies
- **STORMILL 51 (4352)**: 1 anomalies
- **EXXARO LAKESIDE 2 (2134)**: 1 anomalies
- **ELVAN PROPERTY (4306)**: 1 anomalies

### Elliptic Envelope

#### sector:
- **Office**: 55 anomalies
- **Industrial**: 46 anomalies
- **Development**: 1 anomalies

#### property_name:
- **1 HOLWOOD PARK (2205)**: 6 anomalies
- **ROUTE 24 (1842)**: 5 anomalies
- **OXFORD 144 – GPT (2228)**: 4 anomalies
- **GIE 1 PORTIONS 1 & 2 OF ERF308 (2484)**: 3 anomalies
- **VEREENIGING STREET 36 (3005)**: 3 anomalies
- **CONSTANTIA PARK (1813)**: 3 anomalies
- **WOODLANDS OFFICE PARK (3790)**: 3 anomalies
- **144 OXFORD (2148)**: 2 anomalies
- **151 ON 5TH (2113)**: 2 anomalies
- **TRADE CENTRE MOUNT EDGECOMBE (2142)**: 2 anomalies
- **HEALTHCARE PARK (2437)**: 2 anomalies
- **THE PLACE (2476)**: 2 anomalies
- **TRIPARK (2104)**: 2 anomalies
- **ADVOCATES CHAMBERS (3780)**: 2 anomalies
- **MIDWAY PLACE (2440)**: 2 anomalies
- **MENLYN CORNER (2152)**: 2 anomalies
- **HATFIELD GARDENS (1824)**: 1 anomalies
- **FERNTOWERS (4358)**: 1 anomalies
- **PETER PLACE (1933)**: 1 anomalies
- **116 TEAKWOOD ROAD (2154)**: 1 anomalies
- **EPPING 2 (2121)**: 1 anomalies
- **OXFORD CORNER (2907)**: 1 anomalies
- **RIVONIA CROSSING 1 - GPT (2214)**: 1 anomalies
- **WILLOWBRIDGE PLACE (2151)**: 1 anomalies
- **COUNTRY CLUB ESTATE (2509)**: 1 anomalies
- **21 IMPALA ROAD (1885)**: 1 anomalies
- **FOUNTAINS MOTOWN (1917)**: 1 anomalies
- **34 & 36 FRICKER ROAD (3705)**: 1 anomalies
- **IMPALA ROAD (2101)**: 1 anomalies
- **ALBION SPRINGS (3708)**: 1 anomalies
- **EQUITABLE DEVELOPMENT (4308)**: 1 anomalies
- **GILLOOLYS VIEW (1819)**: 1 anomalies
- **200 ON MAIN (2415)**: 1 anomalies
- **FICUS PLACE (3599)**: 1 anomalies
- **11 ADDERLEY (3600)**: 1 anomalies
- **STERLING INDUSTRIAL PARK (2188)**: 1 anomalies
- **MIDRAND CENTRAL BUS PARK 520 (2180)**: 1 anomalies
- **VEREENINGING STREET 36 (3005)**: 1 anomalies
- **28 SACKS CIRCLE (2196)**: 1 anomalies
- **THE OVAL (2445)**: 1 anomalies
- **GALLAGHER PLACE (1919)**: 1 anomalies
- **INANDA RD SPRINGFIELD (2418)**: 1 anomalies
- **PAUL SMIT ANDERBOLT (2185)**: 1 anomalies
- **SPARTICOR (3171)**: 1 anomalies
- **GROWTHPOINT BUSINESS PARK (1934)**: 1 anomalies
- **9 FROSTERLEY CRESCENT (3578)**: 1 anomalies
- **GOLD REEF PARK (3110)**: 1 anomalies
- **GIE 3 PORTION 3 OF ERF 306 (2486)**: 1 anomalies
- **CENTRAL PARK - MIDRAND (1811)**: 1 anomalies
- **GRENVILLE (3125)**: 1 anomalies
- **ANSLOW PARK (NESTLE) (2903)**: 1 anomalies
- **MANDY ROAD (3059)**: 1 anomalies
- **1 NORTH WHARF SQUARE (3781)**: 1 anomalies
- **MOUNT JOY (3027)**: 1 anomalies
- **NESTLE (2433)**: 1 anomalies
- **7 WESSELS ROAD (2914)**: 1 anomalies
- **3012A WMM (3730)**: 1 anomalies
- **SERENADE (3064)**: 1 anomalies
- **33 BREE & 30 WATERKANT (3570)**: 1 anomalies
- **DCD DORBYL BOKSBURG (3535)**: 1 anomalies
- **AFSHIP (3103)**: 1 anomalies
- **LONGKLOOF (3548)**: 1 anomalies
- **SUNNYSIDE OFFICE PARK (1846)**: 1 anomalies
- **AIRPORT VIEW (3167)**: 1 anomalies
- **COVORA (3053)**: 1 anomalies
- **EASTGATE BUSINESS PARK (2405)**: 1 anomalies
- **THE DISTRICT (1894)**: 1 anomalies
- **RIVIERA RD OFFICE PARK (7012)**: 1 anomalies
- **PARAMOUNT PLACE (3559)**: 1 anomalies
- **CHAIN AVENUE (3044)**: 1 anomalies
- **ETON OFFICE PARK (2126)**: 1 anomalies
- **EQUITY HOUSE (2501)**: 1 anomalies
- **DITSELA PLACE (1889)**: 1 anomalies

## Temporal Analysis

### Isolation Forest

#### By Year:
- **2019**: 18 anomalies
- **2020**: 21 anomalies
- **2021**: 16 anomalies
- **2022**: 5 anomalies
- **2023**: 6 anomalies
- **2024**: 2 anomalies
- **2025**: 25 anomalies

#### By Month:
- **Jan**: 93 anomalies

#### By Quarter:
- **Q1**: 93 anomalies

### Lof

#### By Year:
- **2019**: 12 anomalies
- **2020**: 14 anomalies
- **2021**: 6 anomalies
- **2022**: 14 anomalies
- **2023**: 4 anomalies
- **2024**: 1 anomalies
- **2025**: 51 anomalies

#### By Month:
- **Jan**: 102 anomalies

#### By Quarter:
- **Q1**: 102 anomalies

### Elliptic Envelope

#### By Year:
- **2019**: 1 anomalies
- **2020**: 53 anomalies
- **2021**: 19 anomalies
- **2022**: 18 anomalies
- **2024**: 1 anomalies
- **2025**: 10 anomalies

#### By Month:
- **Jan**: 102 anomalies

#### By Quarter:
- **Q1**: 102 anomalies

## Consensus Anomalies (Detected by Multiple Models)
- **Count**: 6 (0.01%)
- **Emissions Statistics**:
  - Mean: 2.95
  - Std: 7.21
  - Min: 0.00
  - Max: 17.65
  - Median: 0.01

### Consensus Anomalies by Source:
- **Electricity generated**: 4 anomalies
- **Paper**: 2 anomalies

### Consensus Anomalies by Time Period:

#### By Year:
- **2020**: 1 anomalies
- **2025**: 5 anomalies

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
