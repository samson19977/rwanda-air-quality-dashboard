# app.py
import os
import sys
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

# ------------- CONFIG -------------
REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site', 'Latitude', 'Longitude']
POLLUTANTS = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
THRESHOLDS = {'SO2': 40, 'CO': 4, 'PM10': 45, 'NO2': 25, 'O3': 100, 'PM2.5': 15}
# be permissive with date parsing (no strict format)
st.set_page_config(layout="wide", page_title="🌍 Rwanda Air Quality Analysis", page_icon="🌍")


# ------------- HELPERS -------------
@st.cache_data(ttl=3600)
def load_and_validate(file_or_path, region_label):
    """
    Accepts either a file-like object (uploaded via Streamlit) or a local path string.
    Returns validated DataFrame (may be empty if invalid).
    """
    if file_or_path is None:
        return pd.DataFrame()

    # read CSV
    try:
        if hasattr(file_or_path, "read"):  # uploaded file
            file_or_path.seek(0)
            df = pd.read_csv(file_or_path)
        elif isinstance(file_or_path, str) and os.path.exists(file_or_path):
            df = pd.read_csv(file_or_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"Error reading {region_label} file: {e}")
        return pd.DataFrame()

    # check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.sidebar.error(f"{region_label}: missing columns: {', '.join(missing)}")
        return pd.DataFrame()

    # parse date (permissive)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Site', 'PM2.5'])  # require these columns
    if df.empty:
        st.sidebar.error(f"{region_label}: no valid rows after cleaning")
        return pd.DataFrame()

    # enrich
    df['Region'] = region_label
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()

    # AQI calculation (PM2.5)
    def calculate_aqi(pm25):
        breakpoints = [
            (0, 12, 0, 50),
            (12, 35.4, 51, 100),
            (35.4, 55.4, 101, 150),
            (55.4, 150.4, 151, 200),
            (150.4, 250.4, 201, 300),
            (250.4, float('inf'), 301, 500),
        ]
        try:
            pm = float(pm25)
        except Exception:
            return None
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= pm <= bp_high:
                return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm - bp_low) + aqi_low
        return None

    df['AQI'] = df['PM2.5'].apply(calculate_aqi).round()
    df['AQI Category'] = pd.cut(
        df['AQI'],
        bins=[0, 50, 100, 150, 200, 300, 500],
        labels=['Good', 'Moderate', 'Unhealthy (Sensitive)', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
    )
    return df


# ------------- MAIN UI -------------
def main():
    st.title("🇷🇼 Rwanda Air Quality Dashboard")
    st.header("📊 Air Quality Overview")

    # Sidebar file upload + fallback to local files
    st.sidebar.header("📂 Data Input")
    st.sidebar.write("Upload CSV files or use the default datasets.")
    city_upload = st.sidebar.file_uploader("Urban Air Quality (Kigali) CSV", type=["csv"])
    rural_upload = st.sidebar.file_uploader("Rural Air Quality CSV", type=["csv"])

    # Always use the provided dataset files if no uploads
    city_input = city_upload if city_upload is not None else "AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv"
    rural_input = rural_upload if rural_upload is not None else "AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv"

    st.sidebar.header("📊 Loading Status")
    city_df = load_and_validate(city_input, "Urban")
    rural_df = load_and_validate(rural_input, "Rural")

    # Show error if no data loaded
    if city_df.empty and rural_df.empty:
        st.error("""
        No valid datasets loaded. Please ensure either:
        1. You've uploaded valid CSV files, OR
        2. The default datasets exist in the app folder:
           - AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv
           - AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv
        """)
        st.stop()

    # Combine available data
    frames = []
    if not city_df.empty:
        frames.append(city_df)
    if not rural_df.empty:
        frames.append(rural_df)
    all_data = pd.concat(frames, ignore_index=True)

    # --- Key metrics ---
    cols = st.columns(5)
    def safe_metric(col, label, value, help_text=None, delta=None):
        try:
            col.metric(label, value, help=help_text, delta=delta)
        except Exception as e:
            col.error(f"Error rendering {label}: {e}")

    safe_metric(cols[0], "Total Records", f"{len(all_data):,}")
    safe_metric(cols[1], "Monitoring Sites", all_data['Site'].nunique())
    date_range = f"{all_data['Date'].min().date()} to {all_data['Date'].max().date()}"
    safe_metric(cols[2], "Date Range", date_range)
    avg_pm25 = f"{all_data['PM2.5'].mean():.1f} µg/m³"
    safe_metric(cols[3], "Avg PM2.5", avg_pm25, "WHO Guideline: 15 µg/m³")
    exceed = (all_data['PM2.5'] > THRESHOLDS['PM2.5']).mean() * 100
    exceed_s = f"{exceed:.1f}%"
    safe_metric(cols[4], "Exceedance Rate", exceed_s)

    # --- AQI Pie chart ---
    st.header("🌡️ Air Quality Index (AQI)")
    fig = px.pie(all_data, names='AQI Category', title='AQI Category Distribution', hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

    # --- Time Series Analysis ---
    st.header("📈 Time Series Analysis")
    pollutant = st.selectbox("Select Pollutant", POLLUTANTS)
    time_resolution = st.selectbox("Time Resolution", ["Daily", "Monthly", "Yearly"])
    
    # Group by selected time resolution
    if time_resolution == "Daily":
        group_col = 'Date'
        ts_data = all_data.groupby(['Region', group_col])[pollutant].mean().reset_index()
    elif time_resolution == "Monthly":
        ts_data = all_data.copy()
        ts_data['YearMonth'] = ts_data['Date'].dt.to_period('M')
        ts_data = ts_data.groupby(['Region', 'YearMonth'])[pollutant].mean().reset_index()
        ts_data['YearMonth'] = ts_data['YearMonth'].astype(str)
    else:  # Yearly
        ts_data = all_data.groupby(['Region', 'Year'])[pollutant].mean().reset_index()
    
    # Create time series plot
    if time_resolution == "Monthly":
        x_col = 'YearMonth'
    elif time_resolution == "Yearly":
        x_col = 'Year'
    else:
        x_col = 'Date'
    
    fig = px.line(
        ts_data,
        x=x_col,
        y=pollutant,
        color='Region',
        title=f'{pollutant} Levels Over Time ({time_resolution})',
        labels={pollutant: f'{pollutant} (µg/m³)'}
    )
    fig.add_hline(y=THRESHOLDS[pollutant], line_dash="dash", line_color="red", annotation_text="WHO Threshold")
    st.plotly_chart(fig, use_container_width=True)

    # --- Spatial Analysis ---
    st.header("🗺️ Spatial Distribution")
    if 'Latitude' in all_data.columns and 'Longitude' in all_data.columns:
        site_data = all_data.groupby(['Site', 'Region', 'Latitude', 'Longitude'])[POLLUTANTS].mean().reset_index()
        fig = px.scatter_mapbox(
            site_data,
            lat="Latitude",
            lon="Longitude",
            color="Region",
            size="PM2.5",
            hover_name="Site",
            hover_data=POLLUTANTS,
            zoom=10,
            height=600,
            title="Pollution Levels by Monitoring Site"
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Spatial data not available (missing Latitude/Longitude columns)")

    # Footer
    st.markdown("---")
    st.markdown(
        f"**Data Source**: Rwanda Environment Management Authority  \n"
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"*For official use only*"
    )


# ------------- ENTRYPOINT -------------
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RUN") or os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_SERVER_ADDRESS"):
        main()
    else:
        port = os.environ.get("PORT", "8501")
        cmd = [sys.executable, "-m", "streamlit", "run", sys.argv[0], "--server.port", port, "--server.address", "0.0.0.0"]
        print("Starting Streamlit with:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Failed to start Streamlit:", e)
            sys.exit(1)
