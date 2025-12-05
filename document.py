from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime

# --- Configuration ---
INPUT_FILE = 'urban_planning_impact_report.csv'
STATS_FILE = 'final_city_master_table.csv'

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Green Air Quality Multiplier (GAQM) - Project Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Load Data
    try:
        df_impact = pd.read_csv(INPUT_FILE)
        df_stats = pd.read_csv(STATS_FILE)
        total_cities = len(df_stats)
    except FileNotFoundError:
        print("Error: CSV files not found. Run the analysis script first.")
        return

    # --- Title Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'R')
    pdf.ln(5)

    # --- 1. Executive Summary ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    summary_text = (
        f"This project analyzed environmental data from {total_cities} global cities to quantify "
        "the impact of urban vegetation on air quality (PM2.5). "
        "By controlling for GDP, Population, and Traffic (NO2), the model identified a "
        "statistically significant 'Green Air Quality Multiplier'."
    )
    pdf.multi_cell(0, 7, summary_text)
    pdf.ln(5)

    # --- 2. Key Statistical Findings ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Key Findings', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    findings = [
        "- Greenness vs. Pollution: A significant negative correlation was found.",
        "- The Multiplier: Increasing green space by 1% reduces PM2.5 by ~1.01%.",
        "- Wealth Effect: GDP is the strongest predictor of clean air.",
        "- Traffic: NO2 levels were controlled to isolate the 'tree effect'."
    ]
    
    for item in findings:
        pdf.cell(0, 7, item, 0, 1)
    pdf.ln(5)

    # --- 3. The 'Problem City' Analysis ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Deviation Analysis (The "Delhi" Factor)', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.multi_cell(0, 7, 
        "The model identified cities where pollution exceeds predictions based on their "
        "greenery and wealth. These deviations indicate external factors (festivals, geography)."
    )
    pdf.ln(3)
    
    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(50, 8, 'City', 1)
    pdf.cell(140, 8, 'Deviation Analysis', 1)
    pdf.ln()
    
    # Table Rows (Top 3 Critical)
    pdf.set_font('Arial', '', 9)
    critical_cities = df_impact[df_impact['Deviation_Analysis'].str.contains("CRITICAL")].head(3)
    
    for index, row in critical_cities.iterrows():
        pdf.cell(50, 8, str(row['City_Name']).title(), 1)
        # Handle long text
        analysis = row['Deviation_Analysis'].replace("CRITICAL: ", "")[:80] + "..."
        pdf.cell(140, 8, analysis, 1)
        pdf.ln()
    pdf.ln(5)

    # --- 4. Top Actionable Recommendations ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '4. Urban Planning Recommendations', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    pdf.multi_cell(0, 7, 
        "Based on pollution profiles (High PM2.5 vs High NO2), the following interventions "
        "are recommended for the most polluted cities in the dataset:"
    )
    pdf.ln(3)
    
    # Table Header
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(40, 8, 'City', 1)
    pdf.cell(40, 8, 'Policy Priority', 1)
    pdf.cell(110, 8, 'Vegetation Type', 1)
    pdf.ln()
    
    # Table Rows (Top 5 Polluted)
    pdf.set_font('Arial', '', 8)
    top_polluted = df_impact.head(5)
    
    for index, row in top_polluted.iterrows():
        pdf.cell(40, 8, str(row['City_Name']).title(), 1)
        pdf.cell(40, 8, str(row['Policy_Priority'])[:20]+"...", 1)
        pdf.cell(110, 8, str(row['Vegetation_Recommendation']), 1)
        pdf.ln()

    # Output
    pdf.output('GAQM_Final_Report.pdf')
    print("PDF Report generated successfully: 'GAQM_Final_Report.pdf'")

if __name__ == "__main__":
    generate_pdf()