import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import statsmodels.api as sm
import requests
import warnings

warnings.filterwarnings("ignore")

# ------------- CONFIG -------------
REQUIRED_COLS = ['Date', 'SO2', 'CO', 'PM10', 'NO2', 'O3', 'PM2.5', 'Site', 'Longitude', 'Latitude']
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

# Data URLs (raw CSV links)
CITY_DATA_URL = "https://raw.githubusercontent.com/samson19977/rwanda-air-quality-dashboard/main/AIR_POLLUTION_IN_KIGALI_FROM_2020_TO_2024.csv"
RURAL_DATA_URL = "https://raw.githubusercontent.com/samson19977/rwanda-air-quality-dashboard/main/AIR_POLLUTION_IN_RURAL_FROM_2020_TO_2024.csv"


# ------------- HELPERS -------------

@st.cache_data(ttl=3600)
def load_data(url, region_label):
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing columns in {region_label} data: {', '.join(missing)}")
            return pd.DataFrame()
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'PM2.5', 'Longitude', 'Latitude'])
        if df.empty:
            st.error(f"No valid data in {region_label} file after cleaning")
            return pd.DataFrame()
        
        df['Region'] = region_label
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['MonthName'] = df['Date'].dt.month_name()
        df['DayOfWeek'] = df['Date'].dt.day_name()
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        return df

    except Exception as e:
        st.error(f"Error loading {region_label} data: {str(e)}")
        return pd.DataFrame()


def show_thresholds():
    with st.expander("ℹ️ Air Quality Threshold Guidelines (WHO Standards)"):
        st.markdown("""
        **Threshold levels used in this dashboard (µg/m³):**
        - **Good**: Below safe level
        - **Moderate**: Between safe and moderate levels
        - **Unhealthy**: Above moderate level
        """)
        threshold_data = []
        for pollutant, levels in THRESHOLDS.items():
            threshold_data.append({
                'Pollutant': pollutant,
                'Safe Level': levels['safe'],
                'Moderate Level': levels['moderate'],
                'Unhealthy Level': levels['unhealthy']
            })
        st.table(pd.DataFrame(threshold_data))


def compute_exceedance(df):
    """Add columns to indicate if pollutant exceeds moderate/unhealthy thresholds."""
    for pollutant in POLLUTANTS:
        df[f'{pollutant}_exceeds_moderate'] = df[pollutant] > THRESHOLDS[pollutant]['moderate']
        df[f'{pollutant}_exceeds_unhealthy'] = df[pollutant] > THRESHOLDS[pollutant]['unhealthy']
    return df


def seasonal_decompose_plot(series, title="Seasonal Decomposition"):
    # Use additive model (default) and daily frequency approximated
    decomposition = sm.tsa.seasonal_decompose(series, model='additive', period=365)
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=900, title_text=title)
    return fig


def create_heatmap(df, pollutant, time_col, agg='mean', title="Heatmap"):
    """Create heatmap of pollutant values by time_col (e.g., hour, day of week) and month"""
    pivot_table = pd.pivot_table(df, values=pollutant, index=time_col, columns='Month', aggfunc=agg)
    pivot_table = pivot_table.sort_index()
    fig = px.imshow(pivot_table,
                    labels=dict(x="Month", y=time_col, color=f"{pollutant} (µg/m³)"),
                    title=title,
                    aspect="auto",
                    color_continuous_scale='Inferno')
    return fig


def train_test_split_timeseries(df, feature_cols, target_col, test_size=0.2):
    """Split dataframe for time series forecasting by date"""
    df = df.sort_values('Date')
    split_idx = int(len(df)*(1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values
    return X_train, X_test, y_train, y_test, train, test


def lstm_model(X_train, y_train, X_test, epochs=20, batch_size=32):
    # reshape for LSTM [samples, time steps, features] - here we use 1 timestep
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test_reshaped).flatten()
    return y_pred


def model_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


# ----------------- APP -------------------

def main():
    st.title("🌍 Rwanda Air Quality Dashboard")
    show_thresholds()

    city_df = load_data(CITY_DATA_URL, "Urban")
    rural_df = load_data(RURAL_DATA_URL, "Rural")

    if city_df.empty and rural_df.empty:
        st.error("Failed to load both datasets.")
        return

    data = pd.concat([df for df in [city_df, rural_df] if not df.empty]).reset_index(drop=True)
    data = compute_exceedance(data)

    st.header("📊 Key Metrics")
    cols = st.columns(4)
    cols[0].metric("Total Records", len(data))
    cols[1].metric("Monitoring Sites", data['Site'].nunique())
    cols[2].metric("Avg PM2.5", f"{data['PM2.5'].mean():.1f} µg/m³",
                  help=f"WHO Moderate Threshold: {THRESHOLDS['PM2.5']['moderate']} µg/m³")
    exceed_pct = (data['PM2.5'] > THRESHOLDS['PM2.5']['moderate']).mean() * 100
    cols[3].metric("Exceedance Rate", f"{exceed_pct:.1f}%",
                  delta=f"{exceed_pct:.1f}% above moderate threshold")

    # ---------------- Spatial Analysis ----------------
    st.header("🗺️ Spatial Pollutant Exceedance Map")

    pollutant_for_map = st.selectbox("Select Pollutant for Spatial Analysis", POLLUTANTS, index=5)

    # Aggregate exceedance by site
    exceedance_by_site = data.groupby(['Site', 'Longitude', 'Latitude'])[f'{pollutant_for_map}_exceeds_moderate'].mean().reset_index()
    exceedance_by_site['Exceedance %'] = exceedance_by_site[f'{pollutant_for_map}_exceeds_moderate'] * 100

    fig_map = px.scatter_mapbox(
        exceedance_by_site,
        lat="Latitude",
        lon="Longitude",
        size="Exceedance %",
        color="Exceedance %",
        color_continuous_scale="Reds",
        size_max=30,
        zoom=7,
        mapbox_style="carto-positron",
        hover_name="Site",
        hover_data={"Exceedance %": ':.1f', "Latitude": False, "Longitude": False},
        title=f"Spatial Exceedance of {pollutant_for_map} Above Moderate Threshold by Site"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ---------------- Advanced Temporal Analysis ----------------
    st.header("📅 Advanced Temporal Analysis")

    tab_seasonal, tab_heatmap, tab_multi_year = st.tabs(["Seasonal Decomposition (PM2.5)", "Pollutant Heatmap", "Multi-Year Comparison"])

    with tab_seasonal:
        st.subheader("Seasonal Decomposition of PM2.5 (Urban + Rural Combined)")
        try:
            # Aggregate daily average PM2.5 for time series decomposition
            daily_pm25 = data.groupby('Date')['PM2.5'].mean().dropna()
            decomposition = sm.tsa.seasonal_decompose(daily_pm25, model='additive', period=365)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'))
            fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'))
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'))
            fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'))
            fig.update_layout(height=600, title="Seasonal Decomposition of PM2.5 (Additive Model)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Seasonal decomposition failed: " + str(e))

    with tab_heatmap:
        st.subheader("Pollutant Levels Heatmap by Day of Week and Month")
        pollutant_heatmap = st.selectbox("Select pollutant for heatmap", POLLUTANTS, index=5)
        heatmap_fig = create_heatmap(data, pollutant_heatmap, 'DayOfWeek', agg='mean',
                                    title=f"Average {pollutant_heatmap} Levels by Day of Week and Month")
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab_multi_year:
        st.subheader("Multi-Year Monthly Average PM2.5 Comparison")
        monthly_avg = data.groupby(['Year', 'Month'])['PM2.5'].mean().reset_index()
        fig_multi = px.line(monthly_avg, x='Month', y='PM2.5', color='Year',
                            labels={'Month': 'Month', 'PM2.5': 'Avg PM2.5 (µg/m³)', 'Year': 'Year'},
                            title="Monthly Average PM2.5 by Year")
        st.plotly_chart(fig_multi, use_container_width=True)

    # -------------- Correlation Heatmap --------------
    st.header("🔗 Pollutant Correlation Heatmap")

    corr = data[POLLUTANTS].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Between Pollutants")
    st.plotly_chart(fig_corr, use_container_width=True)

    # -------------- Exceedance Frequency --------------
    st.header("⚠️ Pollutant Exceedance Frequency by Site")

    exceedance_freq = pd.DataFrame()
    for pollutant in POLLUTANTS:
        temp = data.groupby('Site')[f'{pollutant}_exceeds_moderate'].mean().reset_index()
        temp['Pollutant'] = pollutant
        temp.rename(columns={f'{pollutant}_exceeds_moderate': 'Exceedance Rate'}, inplace=True)
        exceedance_freq = pd.concat([exceedance_freq, temp], axis=0)

    fig_exceed = px.bar(exceedance_freq, x='Site', y='Exceedance Rate', color='Pollutant',
                        title="Pollutant Exceedance Rates Above Moderate Threshold by Site",
                        labels={'Exceedance Rate': 'Rate', 'Site': 'Monitoring Site'},
                        barmode='group')
    st.plotly_chart(fig_exceed, use_container_width=True)

    # -------------- Prediction Model ----------------
    st.header("🤖 Short-term PM2.5 Forecasting per Site")

    selected_site = st.selectbox("Select Monitoring Site for Forecasting", data['Site'].unique())
    site_data = data[data['Site'] == selected_site].sort_values('Date')

    if len(site_data) < 50:
        st.warning("Not enough data points for reliable prediction at this site.")
    else:
        # Features and target
        feature_cols = ['SO2', 'CO', 'PM10', 'NO2', 'O3']
        target_col = 'PM2.5'

        # Clean NAs
        site_data = site_data.dropna(subset=feature_cols + [target_col])

        # Train-test split
        X_train, X_test, y_train, y_test, train_df, test_df = train_test_split_timeseries(site_data, feature_cols, target_col)

        # Models to run
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        }

        # Train & predict traditional models
        preds = {}
        metrics = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[name] = y_pred
            metrics[name] = model_metrics(y_test, y_pred)

        # LSTM model prediction
        try:
            y_pred_lstm = lstm_model(X_train, y_train, X_test, epochs=15)
            preds['LSTM'] = y_pred_lstm
            metrics['LSTM'] = model_metrics(y_test, y_pred_lstm)
        except Exception as e:
            st.warning("LSTM model training failed: " + str(e))

        # Show leaderboard metrics
        metric_df = pd.DataFrame(metrics).T
        metric_df.columns = ['MSE', 'MAE', 'R2']
        metric_df = metric_df.sort_values(by='MSE')
        st.subheader("Model Performance Leaderboard")
        st.dataframe(metric_df.style.format("{:.3f}"))

        # Plot actual vs predicted for top 3 models by MSE
        top_models = metric_df.index[:3]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=test_df['Date'], y=y_test, mode='lines+markers', name='Actual PM2.5'))
        for model_name in top_models:
            fig_pred.add_trace(go.Scatter(x=test_df['Date'], y=preds[model_name], mode='lines', name=f'Predicted - {model_name}'))

        fig_pred.update_layout(title=f"Actual vs Predicted PM2.5 at {selected_site}",
                               xaxis_title="Date",
                               yaxis_title="PM2.5 (µg/m³)",
                               height=600)
        st.plotly_chart(fig_pred, use_container_width=True)

    # -------------- Animated Temporal PM2.5 --------------
    st.header("📈 Animated PM2.5 Temporal Changes Across Sites")

    data_anim = data.dropna(subset=['PM2.5', 'Latitude', 'Longitude'])
    data_anim['Date_str'] = data_anim['Date'].dt.strftime('%Y-%m-%d')

    fig_anim = px.scatter_mapbox(data_anim,
                                lat="Latitude",
                                lon="Longitude",
                                color="PM2.5",
                                size="PM2.5",
                                animation_frame="Date_str",
                                color_continuous_scale="Inferno",
                                size_max=20,
                                zoom=6,
                                mapbox_style="carto-positron",
                                title="Daily PM2.5 Levels Across Rwanda (Animated)")
    st.plotly_chart(fig_anim, use_container_width=True)


if __name__ == "__main__":
    main()
