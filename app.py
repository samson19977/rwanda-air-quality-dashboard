# app.py
import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# ------------- CONFIG -------------
REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site']
POLLUTANTS = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
THRESHOLDS = {'SO2': 40, 'CO': 4, 'PM10': 45, 'NO2': 25, 'O3': 100, 'PM2.5': 15}
st.set_page_config(layout="wide", page_title="🌍 Rwanda Air Quality Analysis", page_icon="🌍")

# ------------- HELPERS -------------
@st.cache_data(ttl=3600)
def load_data(file_path, region_label):
    """Load and validate data from specified file path"""
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns in {region_label} data: {', '.join(missing)}")
            return pd.DataFrame()
        
        # Basic processing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'PM2.5'])
        if df.empty:
            st.error(f"No valid data in {region_label} file after cleaning")
            return pd.DataFrame()
            
        df['Region'] = region_label
        df['Year'] = df['Date'].dt.year
        
        # Simplified AQI calculation
        df['AQI'] = df['PM2.5'].apply(lambda x: 
            50*(x/12) if x <= 12 else
            50 + 50*((x-12)/(35.4-12)) if x <= 35.4 else
            100 + 50*((x-35.4)/(55.4-35.4)) if x <= 55.4 else
            150 + 50*((x-55.4)/(150.4-55.4)) if x <= 150.4 else
            200 + 100*((x-150.4)/(250.4-150.4)) if x <= 250.4 else
            300 + 200*((x-250.4)/(500-250.4))
        ).round()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {region_label} data: {str(e)}")
        return pd.DataFrame()

# ------------- MAIN APP -------------
def main():
    st.title("🇷🇼 Rwanda Air Quality Dashboard")
    
    # Load data - using your specific files
    city_df = load_data("AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv", "Urban")
    rural_df = load_data("AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv", "Rural")
    
    if city_df.empty and rural_df.empty:
        st.error("Failed to load both datasets. Please check the files exist and have the correct format.")
        return
    
    all_data = pd.concat([df for df in [city_df, rural_df] if not df.empty])
    
    # Key Metrics
    st.header("📊 Key Metrics")
    cols = st.columns(3)
    cols[0].metric("Total Records", len(all_data))
    cols[1].metric("Monitoring Sites", all_data['Site'].nunique())
    cols[2].metric("Avg PM2.5", f"{all_data['PM2.5'].mean():.1f} µg/m³")
    
    # Yearly Trends
    st.header("📈 Yearly Pollution Trends")
    pollutant = st.selectbox("Select Pollutant", POLLUTANTS)
    
    yearly_avg = all_data.groupby(['Region', 'Year'])[pollutant].mean().reset_index()
    
    fig = px.line(
        yearly_avg,
        x='Year',
        y=pollutant,
        color='Region',
        title=f'Average {pollutant} Levels by Year',
        labels={pollutant: f'{pollutant} (µg/m³)'}
    )
    fig.add_hline(y=THRESHOLDS[pollutant], line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Summary
    st.header("📋 Pollution Data Summary")
    st.dataframe(
        all_data.groupby('Region')[POLLUTANTS].mean().style
        .format("{:.1f}")
        .highlight_between(axis=0, left=THRESHOLDS, color="#ffcccc"),
        use_container_width=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ------------- ENTRY POINT -------------
if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        # For local execution without Streamlit context
        import streamlit.cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0], "--server.port=8501", "--server.address=0.0.0.0"]
        sys.exit(stcli.main())
