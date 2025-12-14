import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import normaltest
from scipy.stats import chi2_contingency
from scipy import stats
import scikit_posthocs as sp
import streamlit as st
import gdown


# Sidebar Table of Contents
sidebar_title = "TABLE OF CONTENTS"
st.sidebar.title(sidebar_title)
section = st.sidebar.radio('Sections:', [
     "📄 Abstract",
    "🌍 Introduction",
    "🎯 Objectives",
    "🔗 Relevance",
    "⚙️ Methodology",
    "📊 Results",
    "💬 Discussion",
    "✅ Conclusion"

])

# Main Title (always shown)
st.markdown("<h1 style='text-align: center; font-family: Fivo Sans;'>London Bike Sharing: August 2023 Report & Dashboard</h1>", unsafe_allow_html=True)


if section == "📄 Abstract":

    st.markdown("<h3 style='text-align: left;'>Abstract</h3>", unsafe_allow_html=True)

    abstract = 'Climate change and health challenges associated with poor physical fitness have escalated in recent years,' \
    ' prompting the introduction of initiatives to promote sustainable transportation and physical activity.' \
    ' One such initiative is the TfL Cycle Hire scheme in London, which provides publicly accessible bicycles for rent. ' \
    'This study analyses data from the initiative collected in August 2023. The analysis encompasses data wrangling, ' \
    'exploratory data analysis, and hypothesis testing. Key findings reveal that peak demand occurs between 5 pm and 7 pm,' \
    ' most rides last less than 30 minutes, and demand on weekdays is statistically significantly higher than on weekends.'

    st.markdown(f"<p style='text-align: justify, font-family: Candara;'>{abstract}</p>", unsafe_allow_html=True)

if section == "🌍 Introduction":
    st.markdown("<h3 style='text-align: left;'>Introduction</h3>", unsafe_allow_html=True)

    introduction1=  'Air pollution is a major contributor to both public health crises and climate change. According to the WHO, ' \
    'policies aimed at reducing air pollution create a win-win situation for climate and health outcomes. The most common source of air pollution ' \
    'is the emission of carbon dioxide (CO₂). ' \
    ' <a href="https://www.who.int/teams/environment-climate-change-and-health/air-quality-energy-and-health/health-impacts/climate-impacts-of-air-pollution" target="_blank">Air quality, energy and health</a> ' \

    introduction2 =  'Beyond its impact on established health endpoints, air pollution is associated with increased disease incidence and ' \
    'premature mortality. The devastating 1952 air pollution episode in London exemplifies its lethal consequences. Today, air pollution has ' \
    'surpassed poor sanitation and lack of clean drinking water to become the leading environmental cause of premature death. ' \
    ' <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4516868/" target="_blank"> Air pollution and public health: emerging hazards and improved understanding of risk - PMC</a>.'

    introduction3 =  'Strategies to reduce air pollution include transitioning to renewable energy sources such as solar, wind, and hydropower;'\
    ' developing urban forests; and reducing vehicle emissions through alternative transportation options like public transit, walking, and cycling.' \
    ' <a href="https://www.epa.gov/climateimpacts/climate-change-impacts-air-quality" target="_blank"> Climate Change Impacts on Air Quality | US EPA</a>. ' \
    '  Cycling, in particular, offers significant environmental benefits: replacing just one mile of driving with cycling can prevent approximately' \
    ' 300 grams of CO₂ emissions. Additionally, cycling generates considerably less noise pollution than motorised vehicles.' \
    ' <a href="https://antipollutionplan.com/how-does-cycling-reduce-air-pollution.html" target="_blank"> How Does Cycling Reduce Air Pollution</a>.'

    introduction4 =  'Beyond environmental advantages, cycling provides substantial health benefits. Research demonstrates a statistically' \
    ' significant reduction in mortality risk among regular cyclists. Cycling has been shown to lower the risk of cardiovascular disease and' \
    ' type 2 diabetes, making it an effective intervention for both individual health and public welfare. ' \
    ' <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10546027/" target="_blank"> Benefits, risks, barriers, and facilitators to cycling: a narrative review - PMC</a>.'


    st.markdown(f"<p style='text-align: justify;'>{introduction1}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{introduction2}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{introduction3}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{introduction4}</p>", unsafe_allow_html=True)

if section == "🎯 Objectives":
    st.markdown("<h3 style='text-align: justify;'>Objectives</h3>", unsafe_allow_html=True)
    st.markdown('''
        -	Identify the top five most frequented start and end stations based on trip counts.
        -	Determine the most common origin-destination station pairs for trips in August 2023.
        -	Analyse hourly trip distributions to identify peak demand periods throughout the day.
        -	Calculate and compare average trip durations for round trips versus one-way trips.
        -	Assess whether bike trip demand is significantly higher on weekdays than on weekends.
        -	Identify days in August 2023 with the highest average trip durations.
        -	Disaggregate peak hour demand by day to reveal daily temporal usage patterns.
        -	Apply appropriate statistical methods to validate observed patterns and test hypotheses.

    ''')


if section == "🔗 Relevance":
    # Relevance title
    st.markdown("<h3 style='text-align: justify;'>Relevance</h3>", unsafe_allow_html=True)
    # The relevant points of the study
    st.markdown("""
    <div style="text-align: justify;">
    <ul>
        <li>Operational planners can allocate resources efficiently and schedule maintenance during off-peak periods using insights from hourly and daily demand patterns.</li>
        <li>Urban planners can design better cycling infrastructure and improve network connectivity by understanding dominant travel corridors and origin-destination pairs.</li>
        <li>Policymakers can develop evidence-based sustainable transportation policies and promotional strategies informed by temporal usage trends and weekday-weekend demand differences.</li>
        <li>Researchers can advance academic knowledge of shared mobility systems through statistical validation of behavioural patterns and trip type differences.</li>
        <li>Public health officials can support active transportation initiatives that reduce air pollution and improve population health using data-driven evidence of cycling usage patterns.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


if section == "⚙️ Methodology":
    st.markdown("<h3 style='text-align: justify;'>Methodology</h3>", unsafe_allow_html=True)

    # Methodology Paragraphs
    methodology1 = 'The data for this study were obtained from Kaggle and comprise detailed records of 776,527 bicycle journeys' \
    ' from the Transport for London (TfL) Cycle Hire system during August 2023. The TfL Cycle Hire initiative provides publicly ' \
    'accessible bicycles for rent across London, promoting sustainable transportation and physical fitness. This comprehensive ' \
    'dataset captures individual trip-level information, offering a detailed snapshot of cycling activity throughout the month. ' \
    'Key variables include start and end station details, timestamps, bicycle identification numbers, and trip durations.'

    methodology2 = 'The Python programming language was employed to perform all analytical functions. Core libraries such as ‘pandas’ and NumPy ' \
    'were used for data wrangling and feature engineering, while ‘matplotlib’ and ‘seaborn’ supported visualisation. Statistical ' \
    'testing was conducted using SciPy, ensuring reproducibility and methodological rigour throughout the workflow.'

    methodology3 = 'Pre-processing: The pre-processing stage involved cleaning the dataset by removing duplicates and performing feature engineering.'\
    ' Data transformations were applied to convert data types into appropriate formats, ensuring seamless and accurate data analysis.'

    methodology4 = 'Exploratory Analysis: Exploratory analysis was conducted to examine key usage patterns within the dataset. This included identifying ' \
    'the top five origin and destination stations, determining the most popular origin-destination route, analysing peak demand hours,' \
    ' assessing daily demand patterns, and comparing average trip durations between one-way and all-round trips.'

    methodology5 = 'Hypothesis testing: Statistical hypothesis tests were conducted to validate observed patterns and assess relationships within the data.' \
    ' Tests employed included the D\'Agostino-Pearson normality test to assess distribution normality, the Mann-Whitney U test for comparing' \
    ' two independent groups, the Kruskal-Wallis test for comparing multiple groups, and the Chi-square test for examining categorical associations.' 

    # This is the code to display the methodology paragraphs
    st.markdown(f'<p style="text-align: justify;">{methodology1}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: justify;">{methodology2}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: justify;">{methodology3}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: justify;">{methodology4}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: justify;">{methodology5}</p>', unsafe_allow_html=True)

    # This is the code to display the link to the Jupyter Notebook
    jupyter_notebook_link = '<a href= "https://github.com/gekahortor/London-Bike-Sharing-Analysis/blob/2acf01a34fb7e8a24b48fd19800bb58e5bacc5f8/London_bike_sharing_analysis_august_2023_code.ipynb" target="_blank">View the Jupyter Notebook</a>'
    st.markdown(f'<p style="text-align: justify;">{jupyter_notebook_link}</p>', unsafe_allow_html=True)


if section == "📊 Results":
    # This is the code to display the Result title
    st.markdown("<h3 style='text-align: justify;'>Results</h3>", unsafe_allow_html=True)

    # This is the paragraph of the result section
    result1 = 'The most popular origin-destination pair was the route from Hyde Park Corner, Hyde Park to Albert Gate, Hyde Park,' \
    ' accounting for 392 trips. Hyde Park Corner emerged as the most frequented station, serving as both the leading origin and' \
    ' destination point in the bike-sharing network'

    # This is the code to display the first paragraph of the result section
    st.markdown(f'<p style="text-align: justify;">{result1}</p>', unsafe_allow_html=True)

    # ==============================
    # READING THE DATASET
    # ==============================
    # 1. Read the dataset

   

# Fix certificate issue
   # os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

    @st.cache_data
    def load_data():
        file_id = "1TiZLuNseUohhV-0PFbABQAhJwE9t_IyR"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "BikeSharing.csv"
        gdown.download(url, output, quiet=False)
        return pd.read_csv(output)

    BikeSharing = load_data()

    # ==============================
    # CLEANING THE DATASET
    # ==============================

    # 1. Remove duplicate rows
    BikeSharing = BikeSharing.drop_duplicates()

    # 2. Change data types
    #    - Convert date columns to datetime
    BikeSharing[['start_date', 'end_date']] = BikeSharing[['start_date', 'end_date']].apply(pd.to_datetime)

    #    - Convert selected numeric columns to categorical
    BikeSharing[['number',
                'start_station_number',
                'end_station_number',
                'bike_number']] = BikeSharing[['number',
                                                'start_station_number',
                                                'end_station_number',
                                                'bike_number']].astype('category')

    # 3. Remove outliers
    #    - Keep only trips with duration <= 3,600,000 ms (1 hour)
    BikeSharing = BikeSharing[BikeSharing['total_duration_ms'] <= 3600000]

    # 4. Remove leading/trailing whitespaces in station names
    BikeSharing['start_station'] = BikeSharing['start_station'].str.strip()
    BikeSharing['end_station'] = BikeSharing['end_station'].str.strip()

    # ==============================
    # CREATING NEW COLUMNS (FEATURE ENGINEERING)
    # ==============================

    # 1. Total duration in minutes
    BikeSharing['total_duration_min'] = BikeSharing['total_duration_ms'] / (1000 * 60)

    # 2. Trip type (Round trip vs One way trip)
    BikeSharing['trip_type'] = np.where(
        BikeSharing['start_station'] == BikeSharing['end_station'],
        'Round trip',
        'One way trip'
    )

    # 3. Hour of trip start
    BikeSharing['start_hour'] = BikeSharing['start_date'].dt.hour

    # 4. Day of trip start (with ordered weekdays)
    BikeSharing['day_of_start_date'] = BikeSharing['start_date'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    BikeSharing['day_of_start_date'] = pd.Categorical(
        BikeSharing['day_of_start_date'],
        ordered=True,
        categories=weekday_order
    )

    # 5. Week type (Weekday vs Weekend)
    BikeSharing['week_type'] = BikeSharing['day_of_start_date'].apply(
        lambda x: 'Weekday' if x in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] else 'Weekend'
    )

    # 6. Convert start_hour to categorical
    BikeSharing['start_hour'] = BikeSharing['start_hour'].astype('category')

    # 7. Station pairs (start >>> end)
    BikeSharing['station_pairs'] = BikeSharing['start_station'] + ' >>> ' + BikeSharing['end_station']

    # 8. Total duration as timedelta
    BikeSharing['total_duration_time'] = pd.to_timedelta(BikeSharing['total_duration_ms'], unit='ms')


    # ==============================
    # EXPLORATORY ANALYSIS
    # ==============================

    # 1. Identify the top 5 most frequently used start stations
    top5startstations = BikeSharing['start_station'].value_counts().head().index

    # 2. Identify the top 5 most frequently used end stations
    top5endstations = BikeSharing['end_station'].value_counts().head().index

    # 3. Count the number of trips per day of the week (ordered by frequency)
    peakday = BikeSharing['day_of_start_date'].value_counts().index

    # 4. Count the number of trips per trip type (Round trip vs One way trip)
    BikeSharing['trip_type'].value_counts().head()

    # 5. Identify the top 5 bikes with the highest number of trips
    bikeID = BikeSharing['bike_number'].value_counts().head(5).index


    # ==============================
    # COUNTING THE NUMBER OF TRIPS PER STATION PAIR
    # ==============================

    # 1. Create a subset of the dataset containing only one-way trips
    One_way_trip = BikeSharing[BikeSharing['trip_type'] == 'One way trip']

    # 2. Count the number of trips for each station pair (start >>> end)
    #    and extract the top 5 most frequent pairs
    top_station_pairs = One_way_trip['station_pairs'].value_counts().head()

    # 3. Display the result
    st.table(top_station_pairs)


    # ==============================
    # PLOTTING THE TOP 5 START AND END STATIONS
    # ==============================

    st.markdown('<h4 style="text-align: center;">Top 5 Origin and Destination Stations</h4>', unsafe_allow_html=True)

    station_type = st.selectbox("Select a station type:", ["Top 5 Origin Stations", "Top 5 Destination Stations"])

    if station_type == "Top 5 Origin Stations":
        startstation = True
    else:
        startstation = False

    if startstation:

        # 1. Create a figure with two subplots (side by side)
        fig, ax = plt.subplots(1, 1)

        # 2. Plot the top 5 start stations
        countplot = sns.countplot(
            x='start_station',
            color='lightgreen',
            data=BikeSharing[BikeSharing['start_station'].isin(top5startstations)],
            order=top5startstations,
            ax=ax
        )
        ax.set_title('Top 5 Origin Stations', fontdict={
            'fontsize': 16,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        })
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Trips", fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        })
        ax.set_xlabel("Time (hours)", fontdict={
        'fontsize': 14,
        'fontweight': 'bold',
        'fontfamily': 'Candara',
        'color': 'darkgreen'
        }, labelpad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)

    else:

        # 1. Create a figure with two subplots (side by side)
        fig, ax = plt.subplots(1, 1, figsize=(10,7))
        # 3. Plot the top 5 end stations
        sns.countplot(
            x='end_station',
            color='tomato',
            data=BikeSharing[BikeSharing['end_station'].isin(top5endstations)],
            order=top5endstations,
            ax=ax
        )
        ax.set_title('Top 5 Destination Stations', fontdict={
            'fontsize': 16,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkred'
        })
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Trips", fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkred'
        })
        ax.set_xlabel("Time (hours)", fontdict={
        'fontsize': 14,
        'fontweight': 'bold',
        'fontfamily': 'Candara',
        'color': 'darkred'
        }, labelpad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)

    # paragraph 2 of the result section
    result2 = 'Exploratory analysis identified peak demand hours between 17:00 and 19:00 (5 pm to 7 pm). Among weekdays, Tuesday, Wednesday, ' \
    'and Thursday exhibited the highest trip volumes, indicating mid-week concentration of cycling activity.'

    st.markdown(f"<p style='text-align: justify;'>{result2}</p>", unsafe_allow_html=True)

    # ==============================
    # PLOTTING THE NUMBER OF TRIPS PER DAY AND PEAK HOUR OF TRIPS
    # ==============================

    col1, col2  = st.columns(2)


    # 1. Create a figure with two subplots (side by side)
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    # 2. Plot the distribution of trips by hour of the day
    with col1:
        fig1, ax = plt.subplots(1, 1)
        sns.histplot(
            x='start_hour',
            color='lightgreen',
            data=BikeSharing,
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Trips', fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Number of Trips per Hour')
        st.pyplot(fig1)


    # 3. Plot the distribution of trips by day of the week
    with col2:
        fig2, ax = plt.subplots(1, 1)
        sns.countplot(
            y='day_of_start_date',
            color='tomato',
            data=BikeSharing[BikeSharing['day_of_start_date'].isin(peakday)],
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_xlabel('Number of Trips', fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel('Day of Week', fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 
        ax.set_title('Number of Trips per Day')
        st.pyplot(fig2)


    # Paragraph 3 of the result section

    results3  = 'The mean trip duration was 16 minutes for one-way trips and 24 minutes for round trips. Due to positively skewed distributions, ' \
    'median values were used for statistical comparisons: 13 minutes 17 seconds for one-way trips and 24 minutes 35 seconds for round trips.' \
    ' Using the conventional threshold of α = 0.05, the D\'Agostino-Pearson normality test confirmed significant deviation from normality ' \
    '(p < 0.001), justifying the use of non-parametric statistical methods.'

    st.markdown(f"<p style='text-align: justify;'>{results3}</p>", unsafe_allow_html=True)


    st.markdown('<h4 style="text-align: center;">Distribution of Trip Duration by Trip Type</h4>', unsafe_allow_html=True)

    # ==============================
    # PLOTTING A HISTOGRAM TO COMPARE THE DURATION OF EACH TRIP TYPE
    # ==============================

    # 1. Define custom colors for each trip type
    col1, col2 = st.columns(2)

    color1 = col1.color_picker(label='One way trip color', value='#00FF00')
    color2 = col2.color_picker(label='Round trip color', value='#FF0000')

    hue_color = {
        'One way trip': color1,
        'Round trip': color2
    }

    plot_type = st.selectbox("Select a plot type:", ["Histogram", "Violin plot"])

    if plot_type == "Histogram":
        histplot = True
    else:
        histplot = False


    fig , ax = plt.subplots(1, 1, figsize=(10, 7))
    if histplot:
        
        # 2. Plot histogram of trip duration (in minutes), separated by trip type
        sns.histplot(
            x='total_duration_min',
            data=BikeSharing,
            hue='trip_type',
            palette=hue_color,
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_xlabel('Trip Duration (minutes)', fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        }, labelpad=15)
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Trips", fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        })
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)

    else:
        
        # 2. Plot violin plot of trip duration (in minutes), separated by trip type
        sns.violinplot(
            x='total_duration_min',
            data=BikeSharing,
            hue='trip_type',
            palette=hue_color,
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_xlabel('Trip Duration (minutes)', fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        }, labelpad=15)
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
        ax.set_ylabel("Number of Trips", fontdict={
            'fontsize': 14,
            'fontweight': 'bold',
            'fontfamily': 'Candara',
            'color': 'darkgreen'
        })
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)


    # paragraph 4 of the result section
    results4 = 'A Mann-Whitney U test was conducted to determine whether the difference in median trip duration between trip types was ' \
    'statistically significant. The Mann-Whitney U test is a non-parametric test used to compare two independent groups when data violate' \
    ' normality assumptions (Mann-Whitney U test and t-test - Robert Wall Emerson, 2023). The test yielded a U-statistic of 15,025,400,453 ' \
    'and a p-value < 0.001, confirming that the difference in median duration between one-way and round trips was statistically significant.'

    st.markdown(f"<p style='text-align: justify;'>{results4}</p>", unsafe_allow_html=True)

    # paragraph 5 of the result section
    results5 = 'A Chi-square test of independence was conducted to determine whether week type (weekday vs. weekend) influences trip type.' \
    ' Results (χ² = 1124.022, df = 1, p < 0.001) indicated a statistically significant association between week type and trip type. ' \
    'This suggests that travel behaviour differs significantly between weekdays and weekends, with users more likely to take one-way trips' \
    ' on weekdays and all-round trips on weekends.'

    # Displaying the paragraph 5 of the result section
    st.markdown(f"<p style='text-align: justify;'>{results5}</p>", unsafe_allow_html=True)

    # 1. Create a contingency table (crosstab) of trip type vs. week type
    crosstab = pd.crosstab(BikeSharing['trip_type'], BikeSharing['week_type'])
    st.table(crosstab)

    # paragraph 6 of the result section
    result6  = 'An exploratory analysis was conducted to identify peak demand hours for each day of the week.' \
    ' Red bars indicate the hours with the highest demand.'

    # Displaying the paragraph 6 of the result section
    st.markdown(f"<p style='text-align: justify;'>{result6}</p>", unsafe_allow_html=True)

    # ==============================
    # PLOTTING A BAR PLOT OF THE PEAK HOUR OF DEMAND PER DAY
    # ==============================

    # Group by day and hour
    hourly_counts = (
        BikeSharing.groupby(['day_of_start_date', 'start_hour'], observed=False)
        .size()
        .reset_index(name='count')
    )

    # Define weekday order
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hourly_counts['day_of_start_date'] = pd.Categorical(
        hourly_counts['day_of_start_date'],
        categories=weekday_order,
        ordered=True
    )

    # Streamlit UI
    st.markdown('<h4 style="text-align: center;">Peak Hour of Demand per Day</h4>', unsafe_allow_html=True)
    selected_day = st.selectbox("Select a day of the week:", weekday_order)

    # Filter data for selected day
    filtered = hourly_counts[hourly_counts['day_of_start_date'] == selected_day]

    # Sort hours for consistent x-axis
    filtered = filtered.sort_values('start_hour')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = sns.barplot(data=filtered, x='start_hour', y='count', color='lightgreen', ax=ax)

    # Highlight peak hour
    if not filtered.empty:
        peak_idx = filtered['count'].idxmax()
        peak_hour = filtered.loc[peak_idx, 'start_hour']
        for bar in bars.patches:
            if bar.get_x() <= peak_hour < bar.get_x() + bar.get_width():
                bar.set_color('tomato')

    # Labeling
    ax.set_title(f"Trip Counts by Hour on {selected_day}",fontfamily='Candara', fontsize=16, fontweight='bold')
    ax.set_xlabel("Hour of Day", fontfamily='Candara', fontsize=14, fontweight='bold')
    ax.set_ylabel("Trip Count", fontfamily='Candara', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Candara', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

    st.pyplot(fig)

    # paragraph 7 of the result section
    results7 = 'Using the conventional threshold of α = 0.05, p-values less than 0.05 indicate statistically significant differences ' \
    'between days. the Kruskal-Wallis–Wallis test was conducted to determine whether peak hour demand differed significantly across days' \
    ' in the bike-sharing dataset. The results (H = 284.03, p < 0.001) indicate a statistically significant difference in median peak hour ' \
    'demand among the days of the week. This means that at least one day shows a different demand distribution compared to the others, ' \
    'suggesting temporal variability in bike usage patterns across days.'

    # Displaying the paragraph 7 of the result section
    st.markdown(f"<p style='text-align: justify;'>{results7}</p>", unsafe_allow_html=True)

    results8 = 'Further analysis was conducted using Dunn\'s post-hoc test to identify which days differed significantly in peak hour demand.' \
    ' Results revealed that Monday and Sunday exhibited similar demand patterns, while Thursday, Friday, and Saturday formed a cluster with' \
    ' no significant differences among them. All other pairwise comparisons showed statistically significant differences in peak hour demand.'

    # Displaying the paragraph 8 of the result section
    st.markdown(f"<p style='text-align: justify;'>{results8}</p>", unsafe_allow_html=True)

    # ==============================
    # USING DUNN’S TEST TO FIND WHICH SPECIFIC DAYS DIFFER IN PEAK HOUR
    # ==============================

    # 1. Perform Dunn’s post-hoc test after Kruskal-Wallis
    #    - Compares trip start hours across days of the week
    #    - Adjusts p-values using Bonferroni correction
    dunn_results = sp.posthoc_dunn(
        BikeSharing,
        val_col='start_hour',
        group_col='day_of_start_date',
        p_adjust='bonferroni'
    )

    # 2. Display the results (pairwise comparisons between days)
    st.table(dunn_results)


if section == '💬 Discussion':
    # title for discussion section
    st.markdown('<h2 style="text-align: justify;">Discussion</h2>', unsafe_allow_html=True)

    discussion1 = '''
    The majority of origin and destination stations are concentrated around Hyde Park, suggesting that urban planners and policymakers 
    should prioritize infrastructure improvements in this area to accommodate safe and effective cycling. Enhanced bike lanes, signage, 
    and traffic calming measures could support the high volume of users in this corridor.
    '''

    discussion2 = '''
    Operational planners should allocate sufficient bicycles to meet demand during peak hours (5 pm to 7 pm) and on high-demand 
    days (Tuesday, Wednesday, and Thursday). Additionally, implementing robust traffic management strategies during these periods  
    can improve user safety and system efficiency.
    '''

    discussion3 = '''
    Trip duration analysis reveals notable differences between weekdays and weekends. Users predominantly take round trips on weekends,
    suggesting recreational or leisure usage, while weekdays show a preference for one-way trips, indicative of commuting behaviour.
    This insight can inform targeted marketing strategies and pricing structures tailored to different user segments
    '''
    discussion4 = '''
    Further analysis of peak demand hours across individual days revealed distinct temporal patterns. Monday and Sunday exhibit similar
    demand profiles, while Thursday, Friday, and Saturday form a cohesive cluster. These variations suggest that operational strategies
    should be tailored to specific days rather than applying uniform weekday-weekend distinctions.
    '''
    discussion5 = '''
    Future research could identify bicycles involved in longer trips, particularly round trips, to prioritize them for maintenance services.
    Proactive maintenance of high-usage bicycles can prevent damage, reduce downtime, and improve overall system reliability.
    '''


    # Displaying the paragraphs of the discussion section
    st.markdown(f"<p style='text-align: justify;'>{discussion1}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{discussion2}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{discussion3}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{discussion4}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{discussion5}</p>", unsafe_allow_html=True)


if section == '✅ Conclusion':
    # title for conclusion
    st.markdown('<h2 style="text-align: justify;">Conclusion</h2>', unsafe_allow_html=True)

    conclusion1 = '''
    This study analysed 776,527 bicycle journeys from the Transport for London (TfL) Cycle Hire system during August 2023 to understand
    usage patterns and inform sustainable transportation initiatives. The analysis revealed critical insights into spatial demand, temporal 
    usage patterns, and user behaviour that have significant implications for multiple stakeholders.
    '''

    conclusion2 = '''
    Key findings demonstrate that Hyde Park serves as the central hub of cycling activity, with Hyde Park Corner identified as both the
    most frequent origin and destination station. The most popular route connected Hyde Park Corner to Albert Gate, accounting for 392
    trips. Peak demand consistently occurs between 5 pm and 7 pm, with Tuesday, Wednesday, and Thursday experiencing the highest daily 
    trip volumes, indicating strong commuter usage during mid-week periods.
    '''

    conclusion3 = '''
    Statistical analysis confirmed significant behavioural differences between trip types and temporal patterns. Round trips averaged 24 minutes
    compared to 16 minutes for one-way trips, with Mann-Whitney U test results (U = 15,025,400,453, p < 0.001) confirming this difference as 
    statistically significant. Chi-square analysis (χ² = 1124.022, p < 0.001) revealed that trip type preferences differ substantially between
    weekdays and weekends, with users favouring one-way trips during weekdays for commuting and round trips on weekends for leisure activities.
        Further examination of daily demand patterns using Dunn's post-hoc test identified distinct clusters: Monday and Sunday shared similar 
        profiles, while Thursday through Saturday formed a homogeneous group.
    '''

    conclusion4 = '''
    These findings provide actionable insights for urban planners to enhance cycling infrastructure around high-demand areas like Hyde Park,
    enable operational planners to optimise fleet allocation during peak periods, and inform policymakers in developing targeted promotional
    strategies. The evidence supports the role of bike-sharing systems in promoting sustainable transportation and active living, contributing
        to efforts to reduce air pollution and improve public health outcomes. Future research should focus on identifying high-usage bicycles
        for preventive maintenance and exploring seasonal variations in cycling patterns to further optimise system performance and user 
        experience.
    '''

    # Displaying the paragraphs of the conclusion section
    st.markdown(f"<p style='text-align: justify;'>{conclusion1}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{conclusion2}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{conclusion3}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: justify;'>{conclusion4}</p>", unsafe_allow_html=True)




# ==============================
# END OF THE APP

# ==============================


