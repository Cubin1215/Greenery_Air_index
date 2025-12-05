import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- Configuration ---
st.set_page_config(page_title="Green Air Quality Multiplier", layout="wide")
FILE_PATH = 'final_city_master_table.csv'

# --- Load & Prep Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(FILE_PATH)
        # Re-apply log transformations
        df['Log_PM2_5'] = np.log(df['PM2_5_Mean'])
        df['Log_GDP'] = np.log(df['GDP_Capita'])
        df['Log_Population'] = np.log(df['Population'])
        df['Log_NO2'] = np.log(df['NO2_Mean'])
        
        # Title case for display
        if 'Country_Name' in df.columns:
            df['Display_Country'] = df['Country_Name'].str.title()
        else:
            df['Display_Country'] = "Global"
            
        df['Display_City'] = df['City_Name'].str.title()
        
        return df
    except FileNotFoundError:
        return None

def train_model(df):
    formula = "Log_PM2_5 ~ Greenness_Public_Share + Log_NO2 + Log_GDP + Log_Population"
    return ols(formula, data=df).fit()

# --- Helper: Action Plan Generator ---
def get_action_plan(city_data, added_greenery):
    """
    Generates specific recommendations based on city profile.
    Returns a dictionary with lists for each role.
    """
    pm25 = city_data['PM2_5_Mean']
    no2 = city_data['NO2_Mean']
    green = city_data['Greenness_Public_Share']
    
    actions = {
        "Resident": [],
        "Planner": [],
        "Policymaker": []
    }
    
    # 1. PM2.5 Based Actions (Dust/Particulates)
    if pm25 > 35:
        actions["Resident"].append("⚠️ **Health Risk:** Use air purifiers at home and wear N95 masks during high pollution days.")
        actions["Resident"].append("🌱 **Home Greening:** Plant dense shrubs on balconies to trap local dust.")
        actions["Planner"].append("🌳 **Vegetation Barriers:** Plant dense, hairy-leaved species (e.g., Plane, Elm) between roads and sidewalks.")
        actions["Planner"].append("🏗️ **Buffer Zones:** Mandate green buffers around construction sites.")
        actions["Policymaker"].append("🏭 **Industrial Control:** Tighten regulations on solid fuel burning and construction dust.")
    else:
        actions["Resident"].append("✅ Air quality is generally acceptable, but support local green initiatives.")

    # 2. NO2 Based Actions (Traffic)
    if no2 > 30:
        actions["Resident"].append("🚗 **Commute:** Reduce car usage; use public transit or cycle on protected paths.")
        actions["Resident"].append("🏃 **Lifestyle:** Avoid outdoor exercise during rush hours.")
        actions["Planner"].append("🚲 **Infrastructure:** Design separated bike lanes away from direct exhaust fumes.")
        actions["Planner"].append("🌿 **Street Canyon Management:** Avoid creating 'green tunnels' that trap exhaust; allow air flow.")
        actions["Policymaker"].append("🛑 **Traffic Policy:** Implement Low Emission Zones (LEZ) or congestion pricing.")
        actions["Policymaker"].append("⚡ **Fleet:** Incentivize Electric Vehicle (EV) adoption for public transport.")

    # 3. Greenness Based Actions
    if green < 15:
        actions["Planner"].append("🏙️ **Space Optimization:** Focus on 'Pocket Parks' and vertical gardens in dense areas.")
        actions["Policymaker"].append("📜 **3-30-300 Rule:** Adopt the target: 3 trees visible from every home, 30% canopy cover, 300m to nearest park.")
    elif green < 25:
        actions["Planner"].append("🔗 **Connectivity:** Connect existing parks with green corridors.")
    
    # 4. Simulation Specific
    if added_greenery > 0:
        actions["Policymaker"].append(f"🎯 **Target:** Formally adopt a policy goal to increase public green space by {added_greenery}%.")

    return actions

# --- Main UI ---
st.title("🌿 Green Air Quality Planner")
st.markdown("### A Predictive Tool for Urban Vegetation Policy")

df = load_data()

if df is not None:
    # Train Model
    model = train_model(df)
    beta_green = model.params['Greenness_Public_Share']
    r_squared = model.rsquared

    # --- Sidebar ---
    st.sidebar.header("Global Model Stats")
    st.sidebar.metric("Model Confidence (R²)", f"{r_squared:.1%}")
    st.sidebar.info(
        f"**GAQM Coefficient:** {beta_green:.4f}\n\n"
        "This indicates that for every **1% increase** in green space, "
        "pollution decreases by approximately **1.01%**."
    )

    # --- Control Panel ---
    col_controls, col_results = st.columns([1, 2])

    with col_controls:
        st.subheader("📍 Select Location")
        
        countries = sorted(df['Display_Country'].unique())
        selected_country = st.selectbox("Filter by Country", ["All Countries"] + list(countries))
        
        if selected_country != "All Countries":
            city_options = df[df['Display_Country'] == selected_country]
        else:
            city_options = df
            
        selected_city_name = st.selectbox("Select City", city_options['Display_City'].sort_values())
        
        city_subset = df[df['Display_City'] == selected_city_name]
        
        if not city_subset.empty:
            city_data = city_subset.iloc[0]
            
            st.markdown("---")
            st.subheader("🌱 Action Plan")
            added_greenery = st.slider("Add Green Space (%)", 0, 25, 5, help="Simulate increasing the % of public open space.")
        else:
            st.error("City data not found.")
            st.stop()

    with col_results:
        country_display = f", {city_data['Display_Country']}" if 'Display_Country' in city_data else ""
        st.subheader(f"Analysis for {city_data['Display_City']}{country_display}")
        
        # Metrics Row 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Pollution (PM2.5)", f"{city_data['PM2_5_Mean']:.1f} µg/m³")
        c2.metric("Current Greenness", f"{city_data['Greenness_Public_Share']:.1f}%")
        c3.metric("Traffic Level (NO2)", f"{city_data['NO2_Mean']:.1f} µg/m³")
        
        st.markdown("---")
        
        # Prediction Logic
        multiplier_factor = np.exp(beta_green * added_greenery)
        new_pm25 = city_data['PM2_5_Mean'] * multiplier_factor
        reduction_abs = city_data['PM2_5_Mean'] - new_pm25
        reduction_pct = (1 - multiplier_factor) * 100
        
        # Metrics Row 2
        st.markdown(f"#### 🔮 Impact of +{added_greenery}% Greenery")
        p1, p2, p3 = st.columns(3)
        p1.metric("Projected PM2.5", f"{new_pm25:.2f}", delta=f"-{reduction_abs:.2f} µg/m³", delta_color="inverse")
        p2.metric("Relative Improvement", f"{reduction_pct:.2f}%")
        
        # Visualizing the Change
        chart_data = pd.DataFrame({
            "Scenario": ["Current", "Projected"],
            "PM2.5 (µg/m³)": [city_data['PM2_5_Mean'], new_pm25]
        })
        fig = px.bar(chart_data, x="Scenario", y="PM2.5 (µg/m³)", color="Scenario", 
                     color_discrete_map={"Current": "#EF553B", "Projected": "#00CC96"},
                     height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- KEY INSIGHTS (RESTORED OLD SUGGESTIONS) ---
        st.info("💡 **Key Insights & Immediate Actions**")
        
        if city_data['PM2_5_Mean'] > 35:
            st.warning(f"**High PM2.5 Alert:** {city_data['Display_City']} exceeds WHO targets. Prioritize trees with hairy/waxy leaves (e.g., Plane, Elm) to trap dust.")
        
        if city_data['NO2_Mean'] > 30:
            st.warning(f"**High Traffic Alert:** NO2 is high ({city_data['NO2_Mean']:.1f}). Green barriers are needed between roads and pedestrian areas.")
            
        if added_greenery > 0:
            st.success(f"By increasing greenness to **{city_data['Greenness_Public_Share'] + added_greenery:.1f}%**, you could prevent approx. **{reduction_abs:.2f} µg/m³** of particulate exposure.")

        st.markdown("---")

        # --- DETAILED ACTION PLAN (NEW SUGGESTIONS) ---
        st.subheader("📋 Detailed Action Plan")
        
        action_plan = get_action_plan(city_data, added_greenery)
        
        tab1, tab2, tab3 = st.tabs(["👤 For Residents", "🏗️ For Urban Planners", "🏛️ For Policymakers"])
        
        with tab1:
            st.markdown("#### What can you do?")
            for item in action_plan["Resident"]:
                st.write(f"- {item}")
                
        with tab2:
            st.markdown("#### Design Interventions")
            for item in action_plan["Planner"]:
                st.write(f"- {item}")
                
        with tab3:
            st.markdown("#### Policy Decisions")
            for item in action_plan["Policymaker"]:
                st.write(f"- {item}")

else:
    st.error("Data file not found. Please run 'master_data_preparation.py' first.")