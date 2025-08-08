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
        df['Month'] = df['Date'].dt.month
        df['MonthName'] = df['Date'].dt.month_name()
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
        
        # Create threshold table
        threshold_data = []
        for pollutant, levels in THRESHOLDS.items():
            threshold_data.append({
                'Pollutant': pollutant,
                'Safe Level': levels['safe'],
                'Moderate Level': levels['moderate'],
                'Unhealthy Level': levels['unhealthy']
            })
        
        st.table(pd.DataFrame(threshold_data))

# ------------- MAIN APP -------------
def main():
    st.title("Rwanda Air Quality Dashboard")
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
    
    # -------------------------------
    # NEW: Enhanced Pollutant Comparison
    # -------------------------------
    st.header("📊 Pollutant Composition Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Relative Contribution", "PM2.5 Focus", "Distribution Analysis"])
    
    with tab1:
        st.subheader("Pollutant Composition by Region")
        
        # Calculate relative percentages
        region_avg = all_data.groupby('Region')[POLLUTANTS].mean()
        region_pct = region_avg.div(region_avg.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            region_pct.reset_index(),
            x='Region',
            y=POLLUTANTS,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            barmode='stack',
            title='Relative Contribution of Each Pollutant by Region',
            labels={'value': 'Percentage Contribution (%)', 'variable': 'Pollutant'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("PM2.5 Analysis")
        
        cols = st.columns(2)
        with cols[0]:
            # PM2.5 distribution by region
            fig = px.box(
                all_data,
                x='Region',
                y='PM2.5',
                color='Region',
                points="all",
                title='PM2.5 Distribution by Region',
                labels={'PM2.5': 'PM2.5 (µg/m³)'}
            )
            fig.add_hline(
                y=THRESHOLDS['PM2.5']['moderate'],
                line_dash="dash",
                line_color="red",
                annotation_text="Moderate Threshold"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with cols[1]:
            # PM2.5 trends over time
            pm25_trend = all_data.groupby(['Region', 'Year'])['PM2.5'].mean().reset_index()
            fig = px.line(
                pm25_trend,
                x='Year',
                y='PM2.5',
                color='Region',
                title='PM2.5 Annual Trends',
                markers=True,
                labels={'PM2.5': 'PM2.5 (µg/m³)'}
            )
            fig.add_hline(
                y=THRESHOLDS['PM2.5']['moderate'],
                line_dash="dash",
                line_color="red",
                annotation_text="Moderate Threshold"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Pollutant Distribution Analysis")
        
        pollutant = st.selectbox("Select Pollutant to Visualize", POLLUTANTS)
        
        fig = px.violin(
            all_data,
            x='Region',
            y=pollutant,
            color='Region',
            box=True,
            points="all",
            title=f'{pollutant} Distribution by Region',
            labels={pollutant: f'{pollutant} (µg/m³)'}
        )
        fig.add_hline(
            y=THRESHOLDS[pollutant]['moderate'],
            line_dash="dash",
            line_color="red",
            annotation_text="Moderate Threshold"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------
    # Monthly Trends
    # -------------------------------
    st.header("📈 Monthly Pollution Trends")
    pollutant = st.selectbox("Select Pollutant for Trend Analysis", POLLUTANTS, key='monthly_trend')
    
    # Prepare monthly data
    monthly_data = all_data.groupby(['Region', 'Year', 'Month', 'MonthName'])[pollutant].mean().reset_index()
    monthly_data['Year-Month'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
    
    # Create line plot
    fig = px.line(
        monthly_data,
        x='Year-Month',
        y=pollutant,
        color='Region',
        line_group='Region',
        hover_name='MonthName',
        title=f'Monthly {pollutant} Levels Over Time',
        labels={pollutant: f'{pollutant} (µg/m³)', 'Year-Month': 'Time Period'},
        height=500
    )
    
    # Add threshold lines
    fig.add_hline(
        y=THRESHOLDS[pollutant]['moderate'],
        line_dash="dot",
        line_color="orange",
        annotation_text=f"Moderate Threshold ({THRESHOLDS[pollutant]['moderate']} µg/m³)",
        annotation_position="top left"
    )
    
    fig.add_hline(
        y=THRESHOLDS[pollutant]['unhealthy'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Unhealthy Threshold ({THRESHOLDS[pollutant]['unhealthy']} µg/m³)",
        annotation_position="top left"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------
    # Detailed Statistics
    # -------------------------------
    st.header("📋 Detailed Pollution Statistics")
    
    tab1, tab2 = st.tabs(["Regional Averages", "Site-Specific Data"])
    
    with tab1:
        st.subheader("Regional Averages")
        regional_stats = all_data.groupby('Region')[POLLUTANTS].agg(['mean', 'max', 'min']).round(1)
        st.dataframe(regional_stats, use_container_width=True)
    
    with tab2:
        st.subheader("Site-Specific Averages")
        site_stats = all_data.groupby(['Region', 'Site'])[POLLUTANTS].mean().round(1).reset_index()
        st.dataframe(site_stats, use_container_width=True, height=600)
    
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
