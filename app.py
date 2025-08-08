# app.py
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ------------- CONFIG -------------
REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site']
POLLUTANTS = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
THRESHOLDS = {
    'SO2': {'safe': 20, 'moderate': 40, 'unhealthy': 80},
    'CO': {'safe': 2, 'moderate': 4, 'unhealthy': 10},
    'PM10': {'safe': 20, 'moderate': 45, 'unhealthy': 100},
    'NO2': {'safe': 10, 'moderate': 25, 'unhealthy': 50},
    'O3': {'safe': 50, 'moderate': 100, 'unhealthy': 150},
    'PM2.5': {'safe': 10, 'moderate': 15, 'unhealthy': 30}
}
st.set_page_config(layout="wide", page_title="🌍 Rwanda Air Quality Dashboard", page_icon="🌍")

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
        df['Month'] = df['Date'].dt.month_name()
        df['DayOfWeek'] = df['Date'].dt.day_name()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {region_label} data: {str(e)}")
        return pd.DataFrame()

def show_thresholds():
    """Display threshold information in an expandable section"""
    with st.expander("ℹ️ Air Quality Threshold Guidelines (WHO Standards)"):
        st.markdown("""
        **Threshold levels used in this dashboard (µg/m³):**
        - **Good**: Below safe level
        - **Moderate**: Between safe and moderate levels
        - **Unhealthy**: Above moderate level
        """)
        
        # Create threshold table in a way that works with Streamlit
        threshold_data = []
        for pollutant, levels in THRESHOLDS.items():
            threshold_data.append({
                'Pollutant': pollutant,
                'Safe Level': levels['safe'],
                'Moderate Level': levels['moderate'],
                'Unhealthy Level': levels['unhealthy']
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        st.dataframe(threshold_df, use_container_width=True)

# ------------- MAIN APP -------------
def main():
    st.title("🇷🇼 Rwanda Air Quality Dashboard")
    show_thresholds()
    
    # Load data
    city_df = load_data("AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv", "Urban")
    rural_df = load_data("AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv", "Rural")
    
    if city_df.empty and rural_df.empty:
        st.error("Failed to load both datasets. Please check the files exist and have the correct format.")
        return
    
    all_data = pd.concat([df for df in [city_df, rural_df] if not df.empty])
    
    # Key Metrics
    st.header("📊 Key Metrics")
    cols = st.columns(4)
    cols[0].metric("Total Records", len(all_data))
    cols[1].metric("Monitoring Sites", all_data['Site'].nunique())
    cols[2].metric("Avg PM2.5", f"{all_data['PM2.5'].mean():.1f} µg/m³", 
                  help=f"WHO Moderate Threshold: {THRESHOLDS['PM2.5']['moderate']} µg/m³")
    exceed_percent = (all_data['PM2.5'] > THRESHOLDS['PM2.5']['moderate']).mean() * 100
    cols[3].metric("Exceedance Rate", f"{exceed_percent:.1f}%", 
                  delta=f"{exceed_percent:.1f}% above moderate threshold")
    
    # Pollutant Comparison Radar Chart
    st.header("📊 Pollutant Comparison (Relative Levels)")
    avg_pollutants = all_data.groupby('Region')[POLLUTANTS].mean().reset_index()
    
    fig = go.Figure()
    for region in avg_pollutants['Region'].unique():
        region_data = avg_pollutants[avg_pollutants['Region'] == region].iloc[0]
        fig.add_trace(go.Scatterpolar(
            r=region_data[POLLUTANTS].values,
            theta=POLLUTANTS,
            fill='toself',
            name=region,
            hoverinfo='text',
            text=[f"{p}: {v:.1f} µg/m³" for p, v in zip(POLLUTANTS, region_data[POLLUTANTS])]
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly Trends Heatmap
    st.header("🌡️ Monthly Pollution Patterns")
    monthly_data = all_data.groupby(['Region', 'Month', 'Year'])[POLLUTANTS].mean().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=month_order, ordered=True)
    
    pollutant = st.selectbox("Select Pollutant for Monthly Analysis", POLLUTANTS)
    
    heatmap_data = monthly_data.pivot_table(index=['Region', 'Month'], columns='Year', values=pollutant)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color=f"{pollutant} (µg/m³)"),
        aspect="auto",
        color_continuous_scale='RdYlGn_r',
        title=f"Monthly {pollutant} Levels by Region and Year"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Pollution Statistics
    st.header("📋 Detailed Pollution Statistics")
    tab1, tab2 = st.tabs(["By Region", "By Site"])
    
    with tab1:
        st.subheader("Regional Averages")
        regional_stats = all_data.groupby('Region')[POLLUTANTS].agg(['mean', 'max', 'min'])
        st.dataframe(
            regional_stats.style.format("{:.1f}"),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Site-Specific Averages")
        site_stats = all_data.groupby(['Region', 'Site'])[POLLUTANTS].mean().reset_index()
        st.dataframe(
            site_stats.style.format("{:.1f}"),
            use_container_width=True,
            height=600
        )
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                "Data Source: Rwanda Environment Management Authority")

# ------------- ENTRY POINT -------------
if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        import streamlit.cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0], "--server.port=8501", "--server.address=0.0.0.0"]
        sys.exit(stcli.main())
