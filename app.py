# app.py
import os
import sys
import subprocess
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px

# ------------- CONFIG -------------
REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site']
POLLUTANTS = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
THRESHOLDS = {'SO2': 40, 'CO': 4, 'PM10': 45, 'NO2': 25, 'O3': 100, 'PM2.5': 15}
st.set_page_config(layout="wide", page_title="🌍 Rwanda Air Quality Analysis", page_icon="🌍")

# ------------- HELPERS -------------
@st.cache_data(ttl=3600)
def load_and_validate(file_or_path, region_label):
    """Simplified data loading function"""
    if file_or_path is None:
        return pd.DataFrame()

    try:
        if hasattr(file_or_path, "read"):
            file_or_path.seek(0)
            df = pd.read_csv(file_or_path)
        elif isinstance(file_or_path, str) and os.path.exists(file_or_path):
            df = pd.read_csv(file_or_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"Error reading {region_label} file: {e}")
        return pd.DataFrame()

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.sidebar.error(f"{region_label}: missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    # Basic date parsing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'PM2.5'])
    if df.empty:
        st.sidebar.error(f"{region_label}: no valid rows after cleaning")
        return pd.DataFrame()

    # Simplified enrichment - only what we need
    df['Region'] = region_label
    df['Year'] = df['Date'].dt.year
    
    # Basic AQI calculation
    df['AQI'] = df['PM2.5'].apply(lambda x: 
        50*(x/12) if x <= 12 else
        50 + 50*((x-12)/(35.4-12)) if x <= 35.4 else
        100 + 50*((x-35.4)/(55.4-35.4)) if x <= 55.4 else
        150 + 50*((x-55.4)/(150.4-55.4)) if x <= 150.4 else
        200 + 100*((x-150.4)/(250.4-150.4)) if x <= 250.4 else
        300 + 200*((x-250.4)/(500-250.4))
    ).round()
    
    return df

# ------------- MAIN UI -------------
def main():
    st.title("🇷🇼 Rwanda Air Quality Dashboard")
    st.header("📊 Air Quality Overview")

    # Simplified file handling
    st.sidebar.header("📂 Data Input")
    city_input = st.sidebar.file_uploader("Urban Air Quality (Kigali) CSV", type=["csv"]) or "AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv"
    rural_input = st.sidebar.file_uploader("Rural Air Quality CSV", type=["csv"]) or "AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv"

    # Load data
    city_df = load_and_validate(city_input, "Urban")
    rural_df = load_and_validate(rural_input, "Rural")

    if city_df.empty and rural_df.empty:
        st.error("No valid data loaded. Please check your input files.")
        st.stop()

    all_data = pd.concat([df for df in [city_df, rural_df] if not df.empty], ignore_index=True)

    # Key metrics - simplified
    cols = st.columns(3)
    cols[0].metric("Total Records", f"{len(all_data):,}")
    cols[1].metric("Monitoring Sites", all_data['Site'].nunique())
    cols[2].metric("Avg PM2.5", f"{all_data['PM2.5'].mean():.1f} µg/m³")

    # Basic time series plot
    st.header("📈 Pollution Trends")
    pollutant = st.selectbox("Select Pollutant", POLLUTANTS)
    
    # Simplified aggregation by year
    yearly_data = all_data.groupby(['Region', 'Year'])[pollutant].mean().reset_index()
    
    fig = px.line(
        yearly_data,
        x='Year',
        y=pollutant,
        color='Region',
        title=f'Average {pollutant} Levels by Year',
        labels={pollutant: f'{pollutant} (µg/m³)'}
    )
    fig.add_hline(y=THRESHOLDS[pollutant], line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # Basic summary table
    st.header("📋 Data Summary")
    st.dataframe(
        all_data.groupby('Region')[POLLUTANTS].mean().style
        .format("{:.1f}")
        .highlight_between(axis=0, left=THRESHOLDS, color="#ffcccc"),
        use_container_width=True
    )

    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ------------- ENTRYPOINT -------------
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RUN"):
        main()
    else:
        port = os.environ.get("PORT", "8501")
        cmd = [sys.executable, "-m", "streamlit", "run", sys.argv[0], "--server.port", port, "--server.address", "0.0.0.0"]
        subprocess.run(cmd, check=True)
