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
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {region_label} data: {str(e)}")
        return pd.DataFrame()

# ------------- MAIN APP -------------
def main():
    st.title(" Rwanda Air Quality Dashboard")
    
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
    
    # Data Summary - Simplified and more robust
    st.header("📋 Pollution Data Summary")
    try:
        summary_df = all_data.groupby('Region')[POLLUTANTS].mean().round(1)
        st.dataframe(
            summary_df.style.apply(
                lambda x: ['background-color: #ffcccc' if x[col] > THRESHOLDS.get(col, float('inf')) else '' 
                         for col in summary_df.columns],
                axis=1
            ),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not display data summary: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ------------- ENTRY POINT -------------
if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        import streamlit.cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0], "--server.port=8501", "--server.address=0.0.0.0"]
        sys.exit(stcli.main())
