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
        # caller can decide to fallback to local file or sample; here return empty
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


def create_sample_data():
    """Create a small sample dataset so the UI can show something (prevents early exit)."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
    data = {
        'Date': list(dates) * 2,
        'SO2': [5, 10, 2, 6, 4, 8, 3, 7, 9, 6] * 2,
        'CO': [0.3]*20,
        'PM10': [20, 30, 25, 40, 35, 22, 27, 33, 19, 28]*2,
        'NO2': [10]*20,
        'O3': [40]*20,
        'PM2.5': [8, 18, 12, 20, 25, 10, 9, 15, 14, 22]*2,
        'Site': ['Kigali-Center']*10 + ['Rural-1']*10,
        'Latitude': [ -1.95 ]*20,
        'Longitude': [ 30.05 ]*20,
    }
    df = pd.DataFrame(data)
    df['Region'] = ['Urban']*10 + ['Rural']*10
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Hour'] = df['Date'].dt.hour
    df['DayOfWeek'] = df['Date'].dt.day_name()
    # compute AQI like above
    def calc(pm):
        bp = [
            (0, 12, 0, 50),
            (12, 35.4, 51, 100),
            (35.4, 55.4, 101, 150),
            (55.4, 150.4, 151, 200),
            (150.4, 250.4, 201, 300),
            (250.4, float('inf'), 301, 500),
        ]
        for a,b,c,d in bp:
            if a <= pm <= b:
                return ((d - c) / (b - a)) * (pm - a) + c
        return None
    df['AQI'] = df['PM2.5'].apply(calc).round()
    df['AQI Category'] = pd.cut(
        df['AQI'],
        bins=[0,50,100,150,200,300,500],
        labels=['Good','Moderate','Unhealthy (Sensitive)','Unhealthy','Very Unhealthy','Hazardous']
    )
    return df


# ------------- MAIN UI -------------
def main():
    st.title("🇷🇼 Rwanda Air Quality Dashboard")
    st.header("📊 Air Quality Overview")

    # Sidebar file upload + fallback to local files
    st.sidebar.header("📂 Data Input")
    st.sidebar.write("Upload CSV files, or place defaults in the app folder.")
    city_upload = st.sidebar.file_uploader("Urban Air Quality (Kigali) CSV", type=["csv"])
    rural_upload = st.sidebar.file_uploader("Rural Air Quality CSV", type=["csv"])

    # fallback to local filenames if upload not provided
    if city_upload is None and os.path.exists("AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv"):
        city_input = "AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv"
    else:
        city_input = city_upload

    if rural_upload is None and os.path.exists("AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv"):
        rural_input = "AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv"
    else:
        rural_input = rural_upload

    st.sidebar.header("📊 Loading Status")
    city_df = load_and_validate(city_input, "Urban")
    rural_df = load_and_validate(rural_input, "Rural")

    # If both missing/invalid, show a friendly message and use sample data (no st.stop)
    if city_df.empty and rural_df.empty:
        st.warning("No valid datasets loaded (uploads or default files). Showing sample data — upload your CSVs or place them in the app folder to view real data.")
        all_data = create_sample_data()
        using_sample = True
    else:
        # prefer whichever exists; concat existing ones
        frames = []
        if not city_df.empty:
            frames.append(city_df)
        if not rural_df.empty:
            frames.append(rural_df)
        all_data = pd.concat(frames, ignore_index=True)
        using_sample = False

    # --- Key metrics ---
    cols = st.columns(5)
    def safe_metric(col, label, value, help_text=None, delta=None):
        try:
            col.metric(label, value, help=help_text, delta=delta)
        except Exception as e:
            col.error(f"Error rendering {label}: {e}")

    safe_metric(cols[0], "Total Records", f"{len(all_data):,}")
    safe_metric(cols[1], "Monitoring Sites", all_data['Site'].nunique() if 'Site' in all_data.columns else "N/A")
    try:
        date_range = f"{all_data['Date'].min().date()} to {all_data['Date'].max().date()}"
    except Exception:
        date_range = "N/A"
    safe_metric(cols[2], "Date Range", date_range)
    try:
        avg_pm25 = f"{all_data['PM2.5'].mean():.1f} µg/m³"
    except Exception:
        avg_pm25 = "N/A"
    safe_metric(cols[3], "Avg PM2.5", avg_pm25, "WHO Guideline: 15 µg/m³")
    try:
        exceed = (all_data['PM2.5'] > THRESHOLDS['PM2.5']).mean() * 100
        exceed_s = f"{exceed:.1f}%"
    except Exception:
        exceed_s = "N/A"
    safe_metric(cols[4], "Exceedance Rate", exceed_s)

    # --- AQI Pie chart (if available) ---
    if 'AQI Category' in all_data.columns:
        st.header("🌡️ Air Quality Index (AQI)")
        fig = px.pie(all_data, names='AQI Category', title='AQI Category Distribution', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    if using_sample:
        st.info("Displayed dataset: **Sample data** (replace by uploading your CSVs or placing the named CSV files next to app.py).")
    st.markdown(
        f"**Data Source**: Rwanda Environment Management Authority  \n"
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n"
        f"*For official use only*"
    )


# ------------- ENTRYPOINT (makes python app.py work by launching Streamlit) -------------
if __name__ == "__main__":
    # If this process is already a Streamlit server (rare), just call main.
    # Otherwise, try to launch Streamlit CLI so the container actually runs a server.
    # Many deployment platforms call `python app.py` — this handles that case.
    if os.environ.get("STREAMLIT_RUN") or os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_SERVER_ADDRESS"):
        # likely already running under streamlit
        main()
    else:
        # Start Streamlit as subprocess so container binds to expected port/address
        port = os.environ.get("PORT", "8501")
        cmd = [sys.executable, "-m", "streamlit", "run", sys.argv[0], "--server.port", port, "--server.address", "0.0.0.0"]
        print("Starting Streamlit with:", " ".join(cmd))
        try:
            # This blocks (desired) and streams logs to stdout/stderr
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Failed to start Streamlit:", e)
            print("If you prefer, run locally with: streamlit run app.py")
            sys.exit(1)
