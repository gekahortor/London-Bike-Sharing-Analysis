# 🚴‍♂️ Analysis of London Bike Sharing Data (August 2023)
## **Abstract**
<div style="text-align: justify">

Climate change and health challenges associated with poor physical fitness have escalated in recent years, prompting the introduction of initiatives to promote sustainable transportation and physical activity. One such initiative is the TfL Cycle Hire scheme in London, which provides publicly accessible bicycles for rent. This study analyses data from the initiative collected in August 2023. The analysis encompasses data wrangling, exploratory data analysis, and hypothesis testing. Key findings reveal that peak demand occurs between 5 pm and 7 pm, most rides last less than 30 minutes, and demand on weekdays is statistically significantly higher than on weekends.

</div>


## **Introduction**
Air pollution is a major contributor to both public health crises and climate change. According to the WHO, policies aimed at reducing air pollution create a win-win situation for climate and health outcomes. The most common source of air pollution is the emission of carbon dioxide (CO₂).  [(Air quality, energy and health)](https://www.who.int/teams/environment-climate-change-and-health/air-quality-energy-and-health/health-impacts/climate-impacts-of-air-pollution).  Beyond its impact on established health endpoints, air pollution is associated with increased disease incidence and premature mortality. The devastating 1952 air pollution episode in London exemplifies its lethal consequences. Today, air pollution has surpassed poor sanitation and lack of clean drinking water to become the leading environmental cause of premature death. [(Air pollution and public health: emerging hazards and improved understanding of risk - PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4516868/) 

Strategies to reduce air pollution include transitioning to renewable energy sources such as solar, wind, and hydropower; developing urban forests; and reducing vehicle emissions through alternative transportation options like public transit, walking, and cycling. [(Climate Change Impacts on Air Quality | US EPA)](https://www.epa.gov/climateimpacts/climate-change-impacts-air-quality) Cycling, in particular, offers significant environmental benefits: replacing just one mile of driving with cycling can prevent approximately 300 grams of CO₂ emissions. Additionally, cycling generates considerably less noise pollution than motorised vehicles. [(How Does Cycling Reduce Air Pollution](https://antipollutionplan.com/how-does-cycling-reduce-air-pollution.html)


Beyond environmental advantages, cycling provides substantial health benefits. Research demonstrates a statistically significant reduction in mortality risk among regular cyclists. Cycling has been shown to lower the risk of cardiovascular disease and type 2 diabetes, making it an effective intervention for both individual health and public welfare. [(Benefits, risks, barriers, and facilitators to cycling: a narrative review - PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10546027/)

---
The reason for this study is as follows:

## **Objectives**
-	Identify the top five most frequented start and end stations based on trip counts.
-	Determine the most common origin-destination station pairs for trips in August 2023.
-	Analyse hourly trip distributions to identify peak demand periods throughout the day.
-	Calculate and compare average trip durations for round trips versus one-way trips.
-	Assess whether bike trip demand is significantly higher on weekdays than on weekends.
-	Identify days in August 2023 with the highest average trip durations.
-	Disaggregate peak hour demand by day to reveal daily temporal usage patterns.
-	Apply appropriate statistical methods to validate observed patterns and test hypotheses.


---

## **Relevance**
-	Operational planners can allocate resources efficiently and schedule maintenance during off-peak periods using insights from hourly and daily demand patterns.
-	Urban planners can design better cycling infrastructure and improve network connectivity by understanding dominant travel corridors and origin-destination pairs.
-	Policymakers can develop evidence-based sustainable transportation policies and promotional strategies informed by temporal usage trends and weekday-weekend demand differences.
-	Researchers can advance academic knowledge of shared mobility systems through statistical validation of behavioural patterns and trip type differences.
-	Public health officials can support active transportation initiatives that reduce air pollution and improve population health using data-driven evidence of cycling usage patterns.
  

---

## **Methodology**
The data for this study were obtained from Kaggle and comprise detailed records of 776,527 bicycle journeys from the Transport for London (TfL) Cycle Hire system during August 2023. The TfL Cycle Hire initiative provides publicly accessible bicycles for rent across London, promoting sustainable transportation and physical fitness. This comprehensive dataset captures individual trip-level information, offering a detailed snapshot of cycling activity throughout the month. Key variables include start and end station details, timestamps, bicycle identification numbers, and trip durations. 

The Python programming language was employed to perform all analytical functions. Core libraries such as ‘pandas’ and NumPy were used for data wrangling and feature engineering, while ‘matplotlib’ and ‘seaborn’ supported visualisation. Statistical testing was conducted using SciPy, ensuring reproducibility and methodological rigour throughout the workflow.

Pre-processing: The pre-processing stage involved cleaning the dataset by removing duplicates and performing feature engineering. Data transformations were applied to convert data types into appropriate formats, ensuring seamless and accurate data analysis.

Exploratory Analysis: Exploratory analysis was conducted to examine key usage patterns within the dataset. This included identifying the top five origin and destination stations, determining the most popular origin-destination route, analysing peak demand hours, assessing daily demand patterns, and comparing average trip durations between one-way and all-round trips.


Hypothesis testing: Statistical hypothesis tests were conducted to validate observed patterns and assess relationships within the data. Tests employed included the D'Agostino-Pearson normality test to assess distribution normality, the Mann-Whitney U test for comparing two independent groups, the Kruskal-Wallis test for comparing multiple groups, and the Chi-square test for examining categorical associations. 

---

## **Results**

The most popular origin-destination pair was the route from Hyde Park Corner, Hyde Park to Albert Gate, Hyde Park, accounting for 392 trips. Hyde Park Corner emerged as the most frequented station, serving as both the leading origin and destination point in the bike-sharing network.


---
| One-Way Trips | Count |
|----------------|-------|
| Hyde Park Corner, Hyde Park → Albert Gate, Hyde Park | 392 |
| Albert Gate, Hyde Park → Hyde Park Corner, Hyde Park | 379 |
| Ackroyd Drive, Bow → Maplin Street, Mile End | 322 |
| Black Lion Gate, Kensington Gardens → Hyde Park Corner, Hyde Park | 303 |
| Hyde Park Corner, Hyde Park → Triangle Car Park, Hyde Park | 294 |
---

<img src="images/Start_end_station.png" alt="Origin and Destination Stations" width="1000">

Exploratory analysis identified peak demand hours between 17:00 and 19:00 (5 pm to 7 pm). Among weekdays, Tuesday, Wednesday, and Thursday exhibited the highest trip volumes, indicating mid-week concentration of cycling activity.

<img src="images/Start_hour.png" alt="Start hour and demand per day" width="1000">
---

The mean trip duration was 16 minutes for one-way trips and 24 minutes for round trips. Due to positively skewed distributions, median values were used for statistical comparisons: 13 minutes 17 seconds for one-way trips and 24 minutes 35 seconds for round trips. Using the conventional threshold of α = 0.05, the D'Agostino-Pearson normality test confirmed significant deviation from normality (p < 0.001), justifying the use of non-parametric statistical methods.

<img src="images/Histogram.png" alt="Histogram" width="1000">

---

<img src="images/Violin.png" alt="Violin" width="1000">
---

A Mann-Whitney U test was conducted to determine whether the difference in median trip duration between trip types was statistically significant. The Mann-Whitney U test is a non-parametric test used to compare two independent groups when data violate normality assumptions (Mann-Whitney U test and t-test - Robert Wall Emerson, 2023). The test yielded a U-statistic of 15,025,400,453 and a p-value < 0.001, confirming that the difference in median duration between one-way and round trips was statistically significant.

---
A Chi-square test of independence was conducted to determine whether week type (weekday vs. weekend) influences trip type. Results (χ² = 1124.022, df = 1, p < 0.001) indicated a statistically significant association between week type and trip type. This suggests that travel behaviour differs significantly between weekdays and weekends, with users more likely to take one-way trips on weekdays and all-round trips on weekends.


| Week Type | One-way Trips | Round Trips |
|------------|----------------|--------------|
| Weekday | 576,628 | 24,272 |
| Weekend | 139,773 | 8,983 |


---

An exploratory analysis was conducted to identify peak demand hours for each day of the week. Red bars indicate the hours with the highest demand.

<img src="images/Peak_hour.png" alt="peak hour" width="1000">
---

Using the conventional threshold of α = 0.05, p-values less than 0.05 indicate statistically significant differences between days. the Kruskal-Wallis–Wallis test was conducted to determine whether peak hour demand differed significantly across days in the bike-sharing dataset. The results (H = 284.03, p < 0.001) indicate a statistically significant difference in median peak hour demand among the days of the week. This means that at least one day shows a different demand distribution compared to the others, suggesting temporal variability in bike usage patterns across days.

--- 
Further analysis was conducted using Dunn's post-hoc test to identify which days differed significantly in peak hour demand. Results revealed that Monday and Sunday exhibited similar demand patterns, while Thursday, Friday, and Saturday formed a cluster with no significant differences among them. All other pairwise comparisons showed statistically significant differences in peak hour demand.

---

| Days | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|------|---------|----------|------------|-----------|---------|-----------|---------|
| **Monday** | 1 | 6.91E-42 | 1.98E-05 | 1.18E-23 | 3.49E-12 | 3.79E-16 | 1 |
| **Tuesday** | 6.91E-42 | 1 | 9.63E-22 | 0.002 | 1.75E-07 | 0.010 | 5.44E-29 |
| **Wednesday** | 1.98E-05 | 9.63E-22 | 1 | 3.36E-08 | 0.034 | 6.05E-05 | 0.045 |
| **Thursday** | 1.18E-23 | 0.002 | 3.36E-08 | 1 | 0.477 | 1 | 6.25E-15 |
| **Friday** | 3.49E-12 | 1.75E-07 | 0.034 | 0.477 | 1 | 1 | 5.12E-07 |
| **Saturday** | 3.79E-16 | 0.010 | 6.05E-05 | 1 | 1 | 1 | 2.14E-10 |
| **Sunday** | 1 | 5.44E-29 | 0.045 | 6.25E-15 | 5.12E-07 | 2.14E-10 | 1 |

---

## **Discussion**
The majority of origin and destination stations are concentrated around Hyde Park, suggesting that urban planners and policymakers should prioritize infrastructure improvements in this area to accommodate safe and effective cycling. Enhanced bike lanes, signage, and traffic calming measures could support the high volume of users in this corridor.

Operational planners should allocate sufficient bicycles to meet demand during peak hours (5 pm to 7 pm) and on high-demand days (Tuesday, Wednesday, and Thursday). Additionally, implementing robust traffic management strategies during these periods can improve user safety and system efficiency.

Trip duration analysis reveals notable differences between weekdays and weekends. Users predominantly take round trips on weekends, suggesting recreational or leisure usage, while weekdays show a preference for one-way trips, indicative of commuting behaviour. This insight can inform targeted marketing strategies and pricing structures tailored to different user segments.

Further analysis of peak demand hours across individual days revealed distinct temporal patterns. Monday and Sunday exhibit similar demand profiles, while Thursday, Friday, and Saturday form a cohesive cluster. These variations suggest that operational strategies should be tailored to specific days rather than applying uniform weekday-weekend distinctions.

Future research could identify bicycles involved in longer trips, particularly round trips, to prioritize them for maintenance services. Proactive maintenance of high-usage bicycles can prevent damage, reduce downtime, and improve overall system reliability.

---

## **Conclusion**
This study analysed 776,527 bicycle journeys from the Transport for London (TfL) Cycle Hire system during August 2023 to understand usage patterns and inform sustainable transportation initiatives. The analysis revealed critical insights into spatial demand, temporal usage patterns, and user behaviour that have significant implications for multiple stakeholders.

Key findings demonstrate that Hyde Park serves as the central hub of cycling activity, with Hyde Park Corner identified as both the most frequent origin and destination station. The most popular route connected Hyde Park Corner to Albert Gate, accounting for 392 trips. Peak demand consistently occurs between 5 pm and 7 pm, with Tuesday, Wednesday, and Thursday experiencing the highest daily trip volumes, indicating strong commuter usage during mid-week periods.

Statistical analysis confirmed significant behavioural differences between trip types and temporal patterns. Round trips averaged 24 minutes compared to 16 minutes for one-way trips, with Mann-Whitney U test results (U = 15,025,400,453, p < 0.001) confirming this difference as statistically significant. Chi-square analysis (χ² = 1124.022, p < 0.001) revealed that trip type preferences differ substantially between weekdays and weekends, with users favouring one-way trips during weekdays for commuting and round trips on weekends for leisure activities. Further examination of daily demand patterns using Dunn's post-hoc test identified distinct clusters: Monday and Sunday shared similar profiles, while Thursday through Saturday formed a homogeneous group.

These findings provide actionable insights for urban planners to enhance cycling infrastructure around high-demand areas like Hyde Park, enable operational planners to optimise fleet allocation during peak periods, and inform policymakers in developing targeted promotional strategies. The evidence supports the role of bike-sharing systems in promoting sustainable transportation and active living, contributing to efforts to reduce air pollution and improve public health outcomes. Future research should focus on identifying high-usage bicycles for preventive maintenance and exploring seasonal variations in cycling patterns to further optimise system performance and user experience.



---











