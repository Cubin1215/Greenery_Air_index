import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Configuration and Data Setup ---

FILE_PATH = 'final_city_master_table.csv'

# Column Mapping
TARGET_VAR = 'PM2_5_Mean'                
GREENNESS_VAR = 'Greenness_Public_Share' 

CONTROL_VARS = [
    'NO2_Mean',      # Traffic/Industrial proxy
    'GDP_Capita',    # Economic development proxy
    'Population'     # Urban density/size proxy
]

CITY_NAME_COL = 'City_Name'

# --- 1. Load Data & Feature Engineering ---

def load_and_transform_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded. Shape: {df.shape}")
        
        required = [TARGET_VAR, GREENNESS_VAR] + CONTROL_VARS
        df.dropna(subset=required, inplace=True)
        
        # Log Transformations
        df['Log_PM2_5'] = np.log(df[TARGET_VAR])
        df['Log_GDP'] = np.log(df['GDP_Capita'])
        df['Log_Population'] = np.log(df['Population'])
        df['Log_NO2'] = np.log(df['NO2_Mean'])
        
        return df
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found.")
        return None

# --- 2. Multicollinearity Check ---

def check_multicollinearity(df, predictors):
    print(r"\n--- 2. Multicollinearity Check (VIF) ---")
    X = df[predictors].assign(const=1)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data = vif_data[vif_data.Variable != 'const'].sort_values(by='VIF', ascending=False)
    print(vif_data.to_string(index=False))
    return vif_data

# --- 3. Regression Analysis ---

def run_gaqm_model(df):
    print(r"\n--- 3. Regression Analysis (Log-Linear) ---")
    formula = "Log_PM2_5 ~ Greenness_Public_Share + Log_NO2 + Log_GDP + Log_Population"
    print(f"Formula: {formula}")
    model = ols(formula, data=df).fit()
    print(model.summary().as_text())
    return model

# --- 4. Outlier Detection ---

def detect_outliers(df, model):
    print(r"\n--- 4. Outlier Detection ---")
    influence = model.get_influence()
    df['Studentized_Residual'] = influence.resid_studentized_external
    df['Predicted_PM25'] = np.exp(model.fittedvalues)
    
    outliers = df[abs(df['Studentized_Residual']) > 3]
    if not outliers.empty:
        print(f"Found {len(outliers)} outliers.")
    else:
        print("No extreme outliers detected.")

# --- 5. Policy Simulation & Report Generation (NEW) ---

def generate_urban_planning_report(df, model):
    """
    Generates a row-by-row actionable report for urban planners.
    """
    print(r"\n--- 5. Generating Urban Planning Impact Report ---")
    
    # 1. Calculate Policy Simulation (+5% Greenness)
    beta_green = model.params['Greenness_Public_Share']
    multiplier = np.exp(beta_green * 5) # Effect of +5% greenness
    
    # 2. Define Expert Rules for Recommendations
    def recommend_vegetation(row):
        """Recommends tree types based on pollution profile."""
        pm25 = row['PM2_5_Mean']
        no2 = row['NO2_Mean']
        
        recs = []
        if pm25 > 30:
            recs.append("High PM-Trapping Species (Broadleaf/Hairy/Waxy leaves like Plane, Linden, Elm)")
        if no2 > 40:
            recs.append("Traffic-Tolerant Species (Hardy trees like Gleditsia, Maple, Ash)")
        if not recs:
            recs.append("Native Ornamental Species (Focus on biodiversity)")
        return " + ".join(recs)

    def analyze_deviations(row):
        """Checks for external factors based on Residuals."""
        resid = row['Studentized_Residual']
        
        if resid > 2.0:
            return "CRITICAL: Pollution much higher than predicted. CHECK EXTERNAL FACTORS: Local festivals (fireworks), agricultural burning, or geography (valley effect)."
        elif resid < -2.0:
            return "POSITIVE: Pollution lower than predicted. Favorable meteorology or effective local regulations."
        else:
            return "Expected performance consistent with model."

    def recommend_policy(row):
        """Suggests policies based on data gaps."""
        green = row['Greenness_Public_Share']
        no2 = row['NO2_Mean']
        
        policies = []
        if green < 10:
            policies.append("URGENT: Micro-parks & Green Roofs required (Greenness < 10%)")
        elif green < 20:
            policies.append("Improve street tree connectivity")
            
        if no2 > 30:
            policies.append("Implement Low Emission Zones (Traffic is major source)")
        
        if not policies:
            policies.append("Maintain current standards")
            
        return "; ".join(policies)

    # 3. Apply Rules
    report_df = df.copy()
    report_df['Projected_PM25_Drop'] = report_df[TARGET_VAR] - (report_df[TARGET_VAR] * multiplier)
    
    report_df['Vegetation_Recommendation'] = report_df.apply(recommend_vegetation, axis=1)
    report_df['Deviation_Analysis'] = report_df.apply(analyze_deviations, axis=1)
    report_df['Policy_Priority'] = report_df.apply(recommend_policy, axis=1)
    
    # 4. Select Columns for Final Report
    cols = [
        CITY_NAME_COL, 
        TARGET_VAR, 
        'Greenness_Public_Share', 
        'Projected_PM25_Drop',
        'Deviation_Analysis',
        'Vegetation_Recommendation',
        'Policy_Priority'
    ]
    
    final_report = report_df[cols].sort_values(by=TARGET_VAR, ascending=False)
    
    # Save
    output_file = 'urban_planning_impact_report.csv'
    final_report.to_csv(output_file, index=False)
    
    print(f"\nReport Generated: {len(final_report)} cities processed.")
    print(f"Top 3 'Problem Cities' (Action Required):")
    print(final_report.head(3)[[CITY_NAME_COL, 'Deviation_Analysis']].to_string(index=False))
    print(f"\nFull actionable report saved to '{output_file}'")

# --- Main ---

if __name__ == "__main__":
    
    df = load_and_transform_data(FILE_PATH)
    
    if df is not None:
        predictors = ['Greenness_Public_Share', 'Log_NO2', 'Log_GDP', 'Log_Population']
        check_multicollinearity(df, predictors)
        
        model = run_gaqm_model(df)
        detect_outliers(df, model)
        
        # New Report Generation Step
        if model.pvalues['Greenness_Public_Share'] < 0.10: # Looser threshold for generating report insights
            generate_urban_planning_report(df, model)
        else:
            print("Model not significant enough for policy generation.")