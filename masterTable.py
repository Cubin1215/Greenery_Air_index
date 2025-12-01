import pandas as pd
import numpy as np
import unicodedata

# --- Configuration: File Paths and Column Names ---

# 1. Define the actual file names
# IMPORTANT: Save your new WHO data as 'who_air_quality.csv'
AIR_QUALITY_FILE = 'who_air_quality.csv' 
GDP_FILE = 'gdp-per-capita-worldbank.csv'
GREENERY_FILE = 'Greenery_index.csv'

# 2. Define clean, short names for the final master table
FINAL_COLUMN_MAP = {
    'PM2_5_Mean': 'PM2_5_Mean',                  # Mean PM2.5 (Target Variable)
    'PM10_Mean': 'PM10_Mean',                    # NEW: Mean PM10 (Coarse Particulate)
    'NO2_Mean': 'NO2_Mean',                      # NEW: Mean NO2 (Traffic Proxy)
    'GDP_Capita': 'GDP_Capita',                  # GDP per capita
    'Greenness_Public_Share': 'Greenness_Public_Share', # Public open space share (Greenness Index)
    'Population': 'Population',                  # Population from WHO data
    'City_Name': 'City_Name',
    'Country_Name': 'Country_Name',              # New mapping for Country linking
    'Year_AQ': 'Air_Quality_Year'                # Track which year the AQ data is from
}

# --- Helper: Text Normalization ---
def normalize_text(text):
    """
    Converts text to lowercase, strips whitespace, and removes accents.
    Ex: "São Paulo " -> "sao paulo"
    """
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Normalize unicode characters to decompose accents (e.g., 'á' -> 'a' + '´')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

# --- 1. Data Cleaning, Aggregation, and Merging Function ---

def create_master_dataset(df_air_quality, df_gdp, df_greenery):
    """
    Cleans, aggregates, and merges the three datasets into the final master table.
    """
    print("--- 2. Cleaning and Feature Engineering ---")
    
    # --- CRITICAL FIX 1: Robust Name Standardization ---
    print("Standardizing names (lowercase, strip whitespace, remove accents)...")
    
    # 2.1 PROCESS NEW WHO AIR QUALITY DATA
    print("2.1: Processing WHO Air Quality data...")
    
    # Step A: Clean City Name (Remove the /CountryCode part, e.g., "A Coruna/ESP" -> "A Coruna")
    # We split by '/' and take the first part [0]
    df_air_quality['city_clean'] = df_air_quality['city'].astype(str).str.split('/').str[0]
    df_air_quality['city_clean'] = df_air_quality['city_clean'].apply(normalize_text)
    
    # Step B: Ensure Numeric Conversions for ALL Pollutants and Population
    # We use errors='coerce' to turn non-numbers (like "NA" strings) into NaN
    cols_to_numeric = ['pm25_concentration', 'pm10_concentration', 'no2_concentration']
    
    for col in cols_to_numeric:
        if col in df_air_quality.columns:
            df_air_quality[col] = pd.to_numeric(df_air_quality[col], errors='coerce')
        else:
            print(f"WARNING: '{col}' not found in WHO data. Creating empty column.")
            df_air_quality[col] = np.nan

    # Handle Population (Check existence first)
    if 'population' in df_air_quality.columns:
        df_air_quality['population'] = pd.to_numeric(df_air_quality['population'], errors='coerce')
    else:
        print("WARNING: 'population' column not found in WHO data. Creating empty column.")
        df_air_quality['population'] = np.nan

    # Drop rows where PM2.5 is missing (Target variable is mandatory)
    df_air_quality.dropna(subset=['pm25_concentration'], inplace=True)
    
    # Step C: Filter for the LATEST available year for each city
    # Sort by City and Year (Descending) so the latest year is first
    df_air_quality = df_air_quality.sort_values(by=['city_clean', 'year'], ascending=[True, False])
    
    # Drop duplicates, keeping the first (latest) occurrence
    df_agg = df_air_quality.drop_duplicates(subset=['city_clean'], keep='first').copy()
    
    # Rename columns to match our Final Map
    df_agg.rename(columns={
        'city_clean': FINAL_COLUMN_MAP['City_Name'],
        'pm25_concentration': FINAL_COLUMN_MAP['PM2_5_Mean'],
        'pm10_concentration': FINAL_COLUMN_MAP['PM10_Mean'],
        'no2_concentration': FINAL_COLUMN_MAP['NO2_Mean'],
        'population': FINAL_COLUMN_MAP['Population'],
        'year': FINAL_COLUMN_MAP['Year_AQ']
    }, inplace=True)
    
    # Keep relevant columns
    df_agg = df_agg[[
        FINAL_COLUMN_MAP['City_Name'], 
        FINAL_COLUMN_MAP['PM2_5_Mean'], 
        FINAL_COLUMN_MAP['PM10_Mean'],
        FINAL_COLUMN_MAP['NO2_Mean'],
        FINAL_COLUMN_MAP['Population'], 
        FINAL_COLUMN_MAP['Year_AQ']
    ]]

    print(f"WHO Air Quality processed. Unique cities: {df_agg.shape[0]}")

    # 2.2 Cleaning GDP and Greenery Data & Preparing for Merge
    
    # Standardize names for Merge
    df_gdp['Entity'] = df_gdp['Entity'].apply(normalize_text)
    df_greenery['City Name'] = df_greenery['City Name'].apply(normalize_text)
    df_greenery['Country or Territory Name'] = df_greenery['Country or Territory Name'].apply(normalize_text)

    # --- GDP Data Cleaning: Prepare for Country-Level Merge ---
    gdp_col = "GDP per capita, PPP (constant 2021 international $)"
    
    # Find the latest year available for each Country (Entity)
    df_gdp = df_gdp.loc[df_gdp.groupby('Entity')['Year'].idxmax()]
    
    # Rename 'Entity' to 'Country_Name'
    df_gdp.rename(columns={'Entity': FINAL_COLUMN_MAP['Country_Name']}, inplace=True)
    
    # Keep only Country Name and GDP Value
    df_gdp = df_gdp[[FINAL_COLUMN_MAP['Country_Name'], gdp_col]].rename(columns={
        gdp_col: FINAL_COLUMN_MAP['GDP_Capita']
    })
    print("GDP data filtered to latest year for each country.")


    # --- Greenery Data Cleaning: Prepare City and Country columns ---
    greenness_col = 'Average share of the built-up area of cities that is open space for public use for all (%) [a]'
    country_col_raw = 'Country or Territory Name'
    
    # Rename columns
    df_greenery.rename(columns={
        'City Name': FINAL_COLUMN_MAP['City_Name'],
        country_col_raw: FINAL_COLUMN_MAP['Country_Name'], 
        greenness_col: FINAL_COLUMN_MAP['Greenness_Public_Share']
    }, inplace=True)
    
    # Keep City, Country, and Greenness
    df_greenery = df_greenery[[FINAL_COLUMN_MAP['City_Name'], FINAL_COLUMN_MAP['Country_Name'], FINAL_COLUMN_MAP['Greenness_Public_Share']]]

    # --- IMPROVEMENT: Fuzzy City Matching for Recovery ---
    # Many WHO cities are named like "Chicago Naperville..." while Greenery has "Chicago".
    # We try to recover these by checking if the Greenery name is a substring of the WHO name.
    
    print("Attempting to recover unmatched cities via substring matching...")
    who_cities = df_agg[FINAL_COLUMN_MAP['City_Name']].unique()
    green_cities = df_greenery[FINAL_COLUMN_MAP['City_Name']].unique()
    
    # Find Greenery cities that don't have an exact match in WHO
    unmatched_green = [g for g in green_cities if g not in who_cities]
    
    correction_map = {}
    for g_city in unmatched_green:
        # Avoid matching very short names to avoid false positives (e.g. "Bo" in "Bologna")
        if len(g_city) < 4: 
            continue
            
        # Find WHO cities that contain this Greenery city name
        matches = [w for w in who_cities if g_city in w]
        
        # If exactly one match found, we assume it's the correct one
        if len(matches) == 1:
            correction_map[matches[0]] = g_city
            
    print(f"Recovered {len(correction_map)} city matches (e.g., {list(correction_map.keys())[:3]} -> {list(correction_map.values())[:3]})")
    
    # Apply corrections to WHO data
    df_agg[FINAL_COLUMN_MAP['City_Name']] = df_agg[FINAL_COLUMN_MAP['City_Name']].replace(correction_map)


    # 2.3 Merging all DataFrames
    print("\n2.3: Merging dataframes...")
    
    # Step A: Merge Air Quality + Greenery (Matches on CITY Only)
    df_master = df_agg.merge(df_greenery, 
                             on=FINAL_COLUMN_MAP['City_Name'], 
                             how='left') 
    
    rows_after_greenery = len(df_master.dropna(subset=[FINAL_COLUMN_MAP['Greenness_Public_Share']]))
    print(f"Matches after Greenery Merge (City level): {rows_after_greenery}")
    
    # Step B: Merge Master + GDP (Matches on COUNTRY)
    df_master = df_master.merge(df_gdp, 
                                on=FINAL_COLUMN_MAP['Country_Name'], 
                                how='left')

    # 2.4 Final Clean-up
    
    initial_rows = len(df_agg)
    
    # List all final columns we need for regression
    # We require ALL of these to be present for a valid row
    regression_columns = [
        FINAL_COLUMN_MAP['City_Name'],
        FINAL_COLUMN_MAP['PM2_5_Mean'],
        FINAL_COLUMN_MAP['Greenness_Public_Share'],
        FINAL_COLUMN_MAP['GDP_Capita'],
        FINAL_COLUMN_MAP['Population'],
        FINAL_COLUMN_MAP['NO2_Mean'] # We now require NO2 for the model
    ]
    
    # Drop rows where ANY regression variable is missing
    df_master.dropna(subset=regression_columns, inplace=True)
    final_rows = len(df_master)
    
    print(f"Dropped {initial_rows - final_rows} cities due to missing data (missing Greenery, GDP, Population, or NO2).")
    print(f"\nMaster Dataset Complete. Final shape: {df_master.shape}")
    
    return df_master

# --- Main Execution ---

if __name__ == '__main__':
    
    # Step 1: Load Actual Data
    print("--- 1. Loading Actual Raw Data Sources ---")
    
    try:
        # Note: WHO CSVs sometimes use different delimiters (like ';'). 
        # If you get a shape error (e.g., 1 column), try adding delimiter=';' or delimiter='\t'
        df_aq = pd.read_csv(AIR_QUALITY_FILE) 
        print(f"Air Quality Data Loaded: {df_aq.shape} rows.")
    except FileNotFoundError:
        print(f"ERROR: File not found at {AIR_QUALITY_FILE}. Please save your WHO data to this filename.")
        exit()

    try:
        df_gdp_raw = pd.read_csv(GDP_FILE)
        print(f"GDP Data Loaded: {df_gdp_raw.shape} rows.")
    except FileNotFoundError:
        print(f"ERROR: File not found at {GDP_FILE}.")
        exit()

    try:
        df_greenery_raw = pd.read_csv(GREENERY_FILE)
        print(f"Greenery Data Loaded: {df_greenery_raw.shape} rows.\n")
    except FileNotFoundError:
        print(f"ERROR: File not found at {GREENERY_FILE}.")
        exit()
    
    # Step 2: Clean, Aggregate, and Merge
    master_df = create_master_dataset(df_aq, df_gdp_raw, df_greenery_raw)
    
    # Display the first few rows of the final dataset
    print("\n--- Final Master Dataset (First 5 Rows) ---")
    print(master_df.head().to_string(index=False)) 
    
    # Save the master file for the next step
    MASTER_OUTPUT_FILE = 'final_city_master_table.csv'
    master_df.to_csv(MASTER_OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved master dataset to {MASTER_OUTPUT_FILE}.")