import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, normaltest, chi2_contingency, kruskal
import streamlit as st
import gdown

# ==============================
# MAIN TITLE
# ==============================
st.markdown("<h1 style='text-align: center; font-family: Fivo Sans;'>🚴 London Bike Sharing: August 2023 Interactive Dashboard</h1>", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    file_id = "1JHqEFMCnBX0Nu9fhXu0U5_XwxA1H3bpv"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "BikeSharing.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

BikeSharing = load_data()

# ==============================
# CLEANING & FEATURE ENGINEERING
# ==============================
BikeSharing = BikeSharing.drop_duplicates()
BikeSharing[['start_date', 'end_date']] = BikeSharing[['start_date', 'end_date']].apply(pd.to_datetime)
BikeSharing[['number','start_station_number','end_station_number','bike_number']] = BikeSharing[['number','start_station_number','end_station_number','bike_number']].astype('category')
BikeSharing = BikeSharing[BikeSharing['total_duration_ms'] <= 3600000]
BikeSharing['start_station'] = BikeSharing['start_station'].str.strip()
BikeSharing['end_station'] = BikeSharing['end_station'].str.strip()
BikeSharing['total_duration_min'] = BikeSharing['total_duration_ms'] / (1000 * 60)
BikeSharing['trip_type'] = np.where(BikeSharing['start_station'] == BikeSharing['end_station'],'Round trip','One way trip')
BikeSharing['start_hour'] = BikeSharing['start_date'].dt.hour.astype('category')
BikeSharing['day_of_start_date'] = BikeSharing['start_date'].dt.day_name()
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
BikeSharing['day_of_start_date'] = pd.Categorical(BikeSharing['day_of_start_date'], ordered=True, categories=weekday_order)
BikeSharing['week_type'] = BikeSharing['day_of_start_date'].apply(lambda x: 'Weekday' if x in weekday_order[:5] else 'Weekend')
BikeSharing['station_pairs'] = BikeSharing['start_station'] + ' >>> ' + BikeSharing['end_station']

# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("🔍 Filters")
selected_day = st.sidebar.selectbox("Select Day of Week:", ["All"] + weekday_order)
selected_trip_type = st.sidebar.selectbox("Select Trip Type:", ["All","One way trip","Round trip"])
plot_choice = st.sidebar.radio("Choose Plot Type:", ["Trips by Hour","Trips by Day","Trip Duration Distribution","Top Stations","Station Pairs"])

# Apply filters
filtered_data = BikeSharing.copy()
if selected_day != "All":
    filtered_data = filtered_data[filtered_data['day_of_start_date'] == selected_day]
if selected_trip_type != "All":
    filtered_data = filtered_data[filtered_data['trip_type'] == selected_trip_type]

# ==============================
# INTERACTIVE VISUALIZATIONS
# ==============================
if plot_choice == "Trips by Hour":
    st.subheader("📊 Trips by Hour of Day")
    fig, ax = plt.subplots()
    sns.histplot(x='start_hour', data=filtered_data, color='lightgreen', ax=ax)
    st.pyplot(fig)

elif plot_choice == "Trips by Day":
    st.subheader("📊 Trips by Day of Week")
    fig, ax = plt.subplots()
    sns.countplot(y='day_of_start_date', data=filtered_data, color='tomato', ax=ax)
    st.pyplot(fig)

elif plot_choice == "Trip Duration Distribution":
    st.subheader("⏱ Trip Duration Distribution")
    plot_type = st.radio("Select Plot Type:", ["Histogram","Violin"])
    fig, ax = plt.subplots()
    if plot_type == "Histogram":
        sns.histplot(x='total_duration_min', data=filtered_data, hue='trip_type', ax=ax)
    else:
        sns.violinplot(x='trip_type', y='total_duration_min', data=filtered_data, ax=ax)
    st.pyplot(fig)

elif plot_choice == "Top Stations":
    st.subheader("🚉 Top 5 Stations")
    station_type = st.radio("Select Station Type:", ["Origin","Destination"])
    if station_type == "Origin":
        top_stations = filtered_data['start_station'].value_counts().head(5)
        fig, ax = plt.subplots()
        sns.barplot(x=top_stations.index, y=top_stations.values, color='lightgreen', ax=ax)
    else:
        top_stations = filtered_data['end_station'].value_counts().head(5)
        fig, ax = plt.subplots()
        sns.barplot(x=top_stations.index, y=top_stations.values, color='tomato', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif plot_choice == "Station Pairs":
    st.subheader("🔗 Top 5 Station Pairs (One-way Trips)")
    one_way = filtered_data[filtered_data['trip_type'] == 'One way trip']
    top_pairs = one_way['station_pairs'].value_counts().head(5)
    st.table(top_pairs)

# ==============================
# STATISTICAL TESTS
# ==============================
st.sidebar.header("📑 Statistical Tests")
if st.sidebar.button("Run Mann-Whitney U Test (Trip Duration)"):
    one_way = BikeSharing[BikeSharing['trip_type'] == 'One way trip']['total_duration_min']
    round_trip = BikeSharing[BikeSharing['trip_type'] == 'Round trip']['total_duration_min']
    stat, p = mannwhitneyu(one_way, round_trip)
    st.write(f"U-statistic: {stat}, p-value: {p:.4f}")

if st.sidebar.button("Run Chi-Square Test (Trip Type vs Week Type)"):
    crosstab = pd.crosstab(BikeSharing['trip_type'], BikeSharing['week_type'])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    st.write(f"Chi²: {chi2:.2f}, df: {dof}, p-value: {p:.4f}")
    st.table(crosstab)

if st.sidebar.button("Run Kruskal-Wallis Test (Peak Hour Demand Across Days)"):
    grouped = [BikeSharing[BikeSharing['day_of_start_date'] == day]['start_hour'].astype(int) for day in weekday_order]
    stat, p = kruskal(*grouped)
    st.write(f"H-statistic: {stat:.2f}, p-value: {p:.4f}")
