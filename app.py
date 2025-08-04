import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

def main():
    try:
        st.set_page_config(
            layout="wide",
            page_title="🌍 Rwanda Air Quality Analysis",
            page_icon="🌍"
        )

        REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site', 'Latitude', 'Longitude']
        POLLUTANTS = ['SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5']
        THRESHOLDS = {
            'SO2': 40, 'CO': 4, 'PM10': 45,
            'NO2': 25, 'O3': 100, 'PM2.5': 15
        }
        DATE_FORMAT = '%m/%d/%Y %H:%M'

        @st.cache_data(ttl=3600, show_spinner="Loading data...")
        def load_and_validate(file_path, region_label):
            try:
                df = pd.read_csv(file_path)
                st.sidebar.success(f"✅ File loaded: {file_path}")
                
                missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                    return pd.DataFrame()

                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT, errors='raise')
                except Exception as e:
                    st.error(f"❌ Date parsing failed: {str(e)}")
                    return pd.DataFrame()

                df = df.dropna(subset=['Date', 'Site', 'PM2.5'])
                if df.empty:
                    st.error("❌ No valid data remaining after cleaning")
                    return pd.DataFrame()

                df['Region'] = region_label
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Hour'] = df['Date'].dt.hour
                df['DayOfWeek'] = df['Date'].dt.day_name()

                def calculate_aqi(pm25):
                    breakpoints = [
                        (0, 12, 0, 50),
                        (12, 35.4, 51, 100),
                        (35.4, 55.4, 101, 150),
                        (55.4, 150.4, 151, 200),
                        (150.4, 250.4, 201, 300),
                        (250.4, float('inf'), 301, 500)
                    ]
                    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
                        if bp_low <= pm25 <= bp_high:
                            return ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                    return 0

                df['AQI'] = df['PM2.5'].apply(calculate_aqi).round()
                df['AQI Category'] = pd.cut(
                    df['AQI'],
                    bins=[0, 50, 100, 150, 200, 300, 500],
                    labels=['Good', 'Moderate', 'Unhealthy (Sensitive)', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
                )

                return df

            except Exception as e:
                st.error(f"❌ Error loading {file_path}: {str(e)}")
                return pd.DataFrame()

        # Load data
        st.sidebar.header("Data Loading Status")
        city_data = load_and_validate("AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv", "Urban")
        rural_data = load_and_validate("AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv", "Rural")

        if city_data.empty or rural_data.empty:
            st.error("❌ Failed to load datasets")
            st.stop()

        all_data = pd.concat([city_data, rural_data], ignore_index=True)

        # Key Metrics
        st.title("🇷🇼 Rwanda Air Quality Dashboard")
        st.header("📊 Air Quality Overview")
        cols = st.columns(5)

        def safe_metric(col, label, value, help_text=None, delta=None):
            try:
                col.metric(label, value, help=help_text, delta=delta)
            except:
                col.error(f"Error rendering {label}")

        safe_metric(cols[0], "Total Records", f"{len(all_data):,}")
        safe_metric(cols[1], "Monitoring Sites", all_data['Site'].nunique())
        safe_metric(cols[2], "Date Range", f"{all_data['Date'].min().date()} to {all_data['Date'].max().date()}")
        safe_metric(cols[3], "Avg PM2.5", f"{all_data['PM2.5'].mean():.1f} µg/m³", "WHO Guideline: 15 µg/m³")
        exceed = (all_data['PM2.5'] > THRESHOLDS['PM2.5']).mean() * 100
        safe_metric(cols[4], "Exceedance Rate", f"{exceed:.1f}%")

        # AQI Pie Chart
        if 'AQI Category' in all_data.columns:
            st.header("🌡️ Air Quality Index (AQI)")
            col1, col2 = st.columns([3, 1])
            with col1:
                fig = px.pie(all_data, names='AQI Category', title='AQI Category Distribution',
                             color_discrete_sequence=px.colors.diverging.RdYlGn[::-1], hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("""
                ### AQI Categories
                - **0-50**: Good ✅  
                - **51-100**: Moderate 💛  
                - **101-150**: Unhealthy (Sensitive) 🧡  
                - **151-200**: Unhealthy ❤️  
                - **201-300**: Very Unhealthy 💔  
                - **301-500**: Hazardous ☠️
                """)

        # Map Visualization
        st.header("📍 Monitoring Locations")
        map_cols = ['Latitude', 'Longitude', 'Site']
        if all(col in all_data.columns for col in map_cols):
            map_tab1, map_tab2 = st.tabs(["Site Map", "Pollution Heatmap"])
            with map_tab1:
                fig = px.scatter_mapbox(
                    all_data.drop_duplicates('Site'),
                    lat="Latitude", lon="Longitude", hover_name="Site",
                    color="Region", mapbox_style="carto-positron", zoom=7
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            with map_tab2:
                fig = px.density_mapbox(
                    all_data, lat="Latitude", lon="Longitude", z="PM2.5",
                    radius=20, zoom=7, mapbox_style="carto-positron",
                    title="PM2.5 Concentration Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Time Series Trends
        st.header("⏳ Pollution Trends")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            site = st.selectbox("Monitoring Site", all_data['Site'].unique())
        with col2:
            pollutant = st.selectbox("Pollutant", [p for p in POLLUTANTS if p in all_data.columns])
        with col3:
            time_agg = st.selectbox("Time Resolution", ['Raw', 'Daily', 'Monthly'])

        site_data = all_data[all_data['Site'] == site]
        if time_agg == 'Daily':
            site_data = site_data.groupby(pd.Grouper(key='Date', freq='D')).mean(numeric_only=True).reset_index()
        elif time_agg == 'Monthly':
            site_data = site_data.groupby(pd.Grouper(key='Date', freq='M')).mean(numeric_only=True).reset_index()

        fig = px.line(site_data, x='Date', y=pollutant,
                      title=f"{pollutant} at {site} ({time_agg})",
                      labels={pollutant: f"{pollutant} (µg/m³)"})
        if pollutant in THRESHOLDS:
            fig.add_hline(y=THRESHOLDS[pollutant], line_dash="dash", line_color="red", annotation_text="WHO Guideline")
        st.plotly_chart(fig, use_container_width=True)

        # Temporal Patterns
        st.header("🕒 Temporal Patterns")
        tabs = st.tabs(["Hourly", "Weekly", "Monthly"])

        with tabs[0]:
            hourly = all_data.groupby(['Site', 'Hour'])['PM2.5'].mean().reset_index()
            fig = px.line(hourly, x='Hour', y='PM2.5', color='Site', title="Avg PM2.5 by Hour")
            fig.add_hline(y=THRESHOLDS['PM2.5'], line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            weekly = all_data.groupby(['Site', 'DayOfWeek'])['PM2.5'].mean().reset_index()
            weekly['DayOfWeek'] = pd.Categorical(weekly['DayOfWeek'], categories=weekday_order, ordered=True)
            fig = px.bar(weekly.sort_values('DayOfWeek'), x='DayOfWeek', y='PM2.5', color='Site', barmode='group',
                         title="Avg PM2.5 by Day")
            fig.add_hline(y=THRESHOLDS['PM2.5'], line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            monthly = all_data.groupby(['Site', 'Month'])['PM2.5'].mean().reset_index()
            fig = px.line(monthly, x='Month', y='PM2.5', color='Site', title="Avg PM2.5 by Month")
            fig.add_hline(y=THRESHOLDS['PM2.5'], line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        # Pollutant Comparison
        st.header("🔬 Pollutant Comparison")
        available_pollutants = [p for p in POLLUTANTS if p in all_data.columns]
        selected = st.multiselect("Select pollutants", available_pollutants, default=available_pollutants[:2])
        if len(selected) >= 2:
            fig = px.scatter_matrix(all_data, dimensions=selected, color="Region", hover_name="Site")
            st.plotly_chart(fig, use_container_width=True)

        # Footer
        st.markdown("---")
        st.markdown(f"""
        **Data Source**: Rwanda Environment Management Authority  
        **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
        *For official use only*
        """)

    except Exception as e:
        st.error(f"App error: {str(e)}")

if __name__ == "__main__":
    try:
        import sys
        from streamlit.web.cli import main as st_main
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(st_main())
    except Exception as e:
        print(f"Startup error: {str(e)}")
