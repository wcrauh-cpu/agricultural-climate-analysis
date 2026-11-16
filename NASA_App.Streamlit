import streamlit as st
import pandas as pd
import numpy as np
import requests

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ClimateFrontier",
    page_icon="🌎",
    layout="wide"
)


# =====================================================
# NASA POWER CONFIG
# =====================================================

NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

COUNTY_COORDINATES = {

    "Iowa": {
        "Polk (Des Moines)": (41.59, -93.62),
        "Linn (Cedar Rapids)": (41.98, -91.67),
        "Scott (Davenport)": (41.52, -90.58),
        "Johnson (Iowa City)": (41.66, -91.53),
        "Black Hawk (Waterloo)": (42.49, -92.34),
        "Woodbury (Sioux City)": (42.49, -96.39),
        "Dubuque (Dubuque)": (42.50, -90.69),
        "Story (Ames)": (42.03, -93.63)
    },

    "Illinois": {
        "Cook (Chicago)": (41.88, -87.63),
        "DuPage (Wheaton)": (41.86, -88.11),
        "Lake (Waukegan)": (42.36, -87.84),
        "Will (Joliet)": (41.53, -88.08),
        "Kane (Geneva)": (41.89, -88.31),
        "Sangamon (Springfield)": (39.80, -89.65),
        "Peoria (Peoria)": (40.70, -89.59),
        "Champaign (Champaign)": (40.11, -88.24),
        "Winnebago (Rockford)": (42.27, -89.09)
    }

}


# =====================================================
# DATA LOADER
# =====================================================

@st.cache_data(ttl=86400)
def fetch_climate_data(lat, lon):

    params = {
        "parameters":
        "T2M,PRECTOTCORR,GWETROOT,ALLSKY_SFC_SW_DWN,RH2M",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": "20150101",
        "end": "20251231",
        "format": "JSON"
    }

    response = requests.get(
        NASA_API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    if "properties" not in data:
        raise ValueError("NASA POWER response invalid.")

    climate = data["properties"]["parameter"]

    df = pd.DataFrame(climate)

    df.index = pd.to_datetime(
        df.index,
        format="%Y%m%d"
    )

    df = df.reset_index(names="DATE")

    df = df.rename(columns={
        "T2M": "Temperature",
        "PRECTOTCORR": "Precipitation",
        "GWETROOT": "Soil Wetness",
        "ALLSKY_SFC_SW_DWN": "Solar Radiation",
        "RH2M": "Humidity"
    })

    numeric_cols = df.columns.drop("DATE")

    df[numeric_cols] = (
        df[numeric_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace(-999, np.nan)
    )

    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month

    return df


# =====================================================
# DROUGHT MODEL
# =====================================================

def calculate_drought_risk(df):

    latest_year = df["Year"].max()

    current = df[df["Year"] == latest_year]

    avg_temp = current["Temperature"].mean()
    total_precip = current["Precipitation"].sum()
    avg_soil = current["Soil Wetness"].mean()
    avg_humidity = current["Humidity"].mean()

    score = 0

    if avg_temp > 25:
        score += 30
    elif avg_temp > 20:
        score += 15

    if total_precip < 500:
        score += 30
    elif total_precip < 800:
        score += 15

    if avg_soil < 0.30:
        score += 25
    elif avg_soil < 0.50:
        score += 10

    if avg_humidity < 50:
        score += 15
    elif avg_humidity < 65:
        score += 5

    score = min(score, 100)

    if score >= 75:
        level = "Extreme"
        color = "#E74C3C"

    elif score >= 50:
        level = "High"
        color = "#E67E22"

    elif score >= 25:
        level = "Moderate"
        color = "#F1C40F"

    else:
        level = "Low"
        color = "#2ECC71"

    return score, level, color


# =====================================================
# CLIMATE STORY
# =====================================================

def generate_story(df):

    annual = (
        df.groupby("Year")["Temperature"]
        .mean()
        .reset_index()
    )

    if len(annual) < 2:
        return "Insufficient historical data."

    change = (
        annual.iloc[-1]["Temperature"]
        - annual.iloc[0]["Temperature"]
    )

    if change > 1:
        return (
            f"Average temperature increased by "
            f"{change:.2f}°C since 2015. "
            "Warming trends can increase crop stress and irrigation demand."
        )

    if change < -1:
        return (
            f"Average temperature decreased by "
            f"{abs(change):.2f}°C since 2015."
        )

    return "Climate has remained relatively stable."


# =====================================================
# HEADER
# =====================================================

st.title("🌎 ClimateFrontier")

st.markdown("""
### NASA POWER Climate Intelligence Dashboard

Analyze climate patterns, drought risk,
and agricultural resilience using NASA POWER data.
""")


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Location")

selected_state = st.sidebar.selectbox(
    "State",
    list(COUNTY_COORDINATES.keys())
)

selected_county = st.sidebar.selectbox(
    "County",
    list(COUNTY_COORDINATES[selected_state].keys())
)

selected_metric = st.sidebar.selectbox(
    "Metric",
    [
        "Temperature",
        "Precipitation",
        "Soil Wetness",
        "Solar Radiation",
        "Humidity"
    ]
)


# =====================================================
# LOAD DATA
# =====================================================

lat, lon = COUNTY_COORDINATES[selected_state][selected_county]

try:

    climate_df = fetch_climate_data(
        lat,
        lon
    )

except Exception as e:

    st.error(f"NASA POWER API Error: {e}")
    st.stop()


# =====================================================
# KPI CARDS
# =====================================================

latest_year = climate_df["Year"].max()

latest = climate_df[
    climate_df["Year"] == latest_year
]

temp = latest["Temperature"].mean()
precip = latest["Precipitation"].sum()
humidity = latest["Humidity"].mean()
soil = latest["Soil Wetness"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg Temp (°C)",
    round(temp, 1)
)

col2.metric(
    "Annual Rainfall (mm)",
    round(precip, 0)
)

col3.metric(
    "Humidity (%)",
    round(humidity, 1)
)

col4.metric(
    "Soil Wetness",
    round(soil, 2)
)


# =====================================================
# DROUGHT RISK
# =====================================================

st.markdown("---")
st.header("🚨 Drought Intelligence")

risk_score, risk_level, risk_color = (
    calculate_drought_risk(climate_df)
)

a, b = st.columns([1, 2])

with a:
    st.metric(
        "Drought Score",
        f"{risk_score}/100"
    )

with b:

    st.markdown(
        f"""
        <div style="
        background:{risk_color};
        color:white;
        padding:20px;
        text-align:center;
        border-radius:10px;
        font-size:24px;
        font-weight:bold;">
        {risk_level.upper()} RISK
        </div>
        """,
        unsafe_allow_html=True
    )

gauge = go.Figure(

    go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Drought Severity Index"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FC3D21"},
            "steps": [
                {"range": [0, 25], "color": "#2ECC71"},
                {"range": [25, 50], "color": "#F1C40F"},
                {"range": [50, 75], "color": "#E67E22"},
                {"range": [75, 100], "color": "#E74C3C"}
            ]
        }
    )
)

st.plotly_chart(
    gauge,
    use_container_width=True
)


# =====================================================
# TREND CHART
# =====================================================

st.markdown("---")
st.header("📈 Historical Trends")

fig = px.line(
    climate_df,
    x="DATE",
    y=selected_metric,
    title=f"{selected_metric} Trend"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =====================================================
# MONTHLY PATTERNS
# =====================================================

st.markdown("---")
st.header("📅 Monthly Climate Pattern")

monthly = (
    climate_df
    .groupby("Month")[selected_metric]
    .mean()
    .reset_index()
)

monthly_fig = px.bar(
    monthly,
    x="Month",
    y=selected_metric,
    color=selected_metric
)

st.plotly_chart(
    monthly_fig,
    use_container_width=True
)


# =====================================================
# CLIMATE STORY
# =====================================================

st.markdown("---")
st.header("🌱 Climate Story")

st.info(
    generate_story(climate_df)
)


# =====================================================
# FORECAST
# =====================================================

st.markdown("---")
st.header("🔮 Temperature Outlook")

yearly = (
    climate_df
    .groupby("Year")["Temperature"]
    .mean()
    .reset_index()
)

if len(yearly) >= 2:

    model = LinearRegression()

    model.fit(
        yearly[["Year"]],
        yearly["Temperature"]
    )

    future = pd.DataFrame({
        "Year": [2026, 2027, 2028, 2029, 2030]
    })

    future["Projected Temperature"] = model.predict(
        future[["Year"]]
    )

    forecast_fig = px.line(
        future,
        x="Year",
        y="Projected Temperature",
        markers=True
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

else:

    st.warning(
        "Not enough data available for prediction."
    )


# =====================================================
# STATE COMPARISON
# =====================================================

st.markdown("---")
st.header("⚖ Iowa vs Illinois")

comparison_results = []

for state_name, counties in COUNTY_COORDINATES.items():

    temperatures = []

    for county_name, coords in counties.items():

        lat2, lon2 = coords

        try:

            temp_df = fetch_climate_data(
                lat2,
                lon2
            )

            temperatures.append(
                temp_df["Temperature"].mean()
            )

        except Exception:
            continue

    if temperatures:

        comparison_results.append({

            "State": state_name,
            "Mean Temperature": np.mean(temperatures)

        })

comparison_df = pd.DataFrame(
    comparison_results
)

comparison_fig = px.bar(
    comparison_df,
    x="State",
    y="Mean Temperature",
    color="State",
    text_auto=".2f"
)

st.plotly_chart(
    comparison_fig,
    use_container_width=True
)


# =====================================================
# MAP
# =====================================================

st.markdown("---")
st.header("🗺 Included Locations")

map_points = []

for state_name, counties in COUNTY_COORDINATES.items():

    for county_name, coords in counties.items():

        map_points.append({
            "County": county_name,
            "State": state_name,
            "lat": coords[0],
            "lon": coords[1]
        })

map_df = pd.DataFrame(map_points)

st.map(map_df)


# =====================================================
# RAW DATA
# =====================================================

st.markdown("---")
st.header("📊 Raw NASA Climate Data")

st.dataframe(
    climate_df,
    use_container_width=True
)


# =====================================================
# DOWNLOAD
# =====================================================

safe_county = (
    selected_county
    .replace("(", "")
    .replace(")", "")
)

csv = climate_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name=f"{safe_county}_climate_data.csv",
    mime="text/csv"
)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
"""
ClimateFrontier uses NASA POWER climate data to help
communities understand climate risks, drought conditions,
and agricultural resilience.
"""
)
