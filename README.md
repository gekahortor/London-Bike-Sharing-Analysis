# 🚲 London Bike Sharing Analysis — August 2023

## 📄 Abstract
Climate change and health challenges linked to poor physical fitness have intensified in recent years.  
The **Transport for London (TfL) Cycle Hire scheme** provides publicly accessible bicycles to encourage sustainable transportation and physical activity.  
This project analyzes **776,527 trips recorded in August 2023**, applying data wrangling, exploratory data analysis, and hypothesis testing.  
Key findings show:
- Peak demand between **5 pm and 7 pm**  
- Most rides lasting **under 30 minutes**  
- **Weekday demand significantly higher** than weekends  

---

## 🌍 Introduction
Air pollution is a leading environmental cause of premature death, surpassing poor sanitation and unsafe drinking water.  
Cycling offers dual benefits:
- **Environmental**: reduces CO₂ emissions (~300g per mile replaced), lowers noise pollution.  
- **Health**: lowers risks of cardiovascular disease and type 2 diabetes, reduces mortality among regular cyclists.  

The TfL Cycle Hire scheme provides a valuable dataset to study shared mobility systems and their impact on sustainability and public health.

---

## 🎯 Objectives
- Identify the **top five start and end stations**.  
- Determine the most common **origin-destination pairs**.  
- Analyze **hourly trip distributions** to detect peak demand.  
- Compare **trip durations** for round trips vs. one-way trips.  
- Assess **weekday vs. weekend demand differences**.  
- Highlight days with the **longest average trip durations**.  
- Apply statistical tests to validate observed patterns.  

---

## ⚙️ Methodology
- **Dataset**: 776,527 trips from TfL Cycle Hire (August 2023), sourced from Kaggle.  
- **Tools**: Python libraries — pandas, NumPy, matplotlib, seaborn, SciPy, scikit-posthocs.  
- **Steps**:
  1. **Pre-processing**: cleaning duplicates, removing outliers, type conversions.  
  2. **Feature engineering**: trip type, duration, station pairs, weekday/weekend classification.  
  3. **Exploratory analysis**: top stations, peak hours, trip durations.  
  4. **Hypothesis testing**:  
     - D’Agostino-Pearson normality test  
     - Mann-Whitney U test  
     - Kruskal-Wallis test  
     - Chi-square test  

---

## 📊 Results
- **Most popular route**: Hyde Park Corner → Albert Gate (392 trips).  
- **Peak demand**: 5–7 pm, especially mid-week (Tue–Thu).  
- **Trip durations**:  
  - Median 13 min (one-way)  
  - Median 24 min (round trip)  
- **Statistical findings**:  
  - Mann-Whitney U confirmed significant differences in trip durations.  
  - Chi-square showed weekday vs. weekend differences in trip type.  
  - Kruskal-Wallis revealed variability in peak-hour demand across days.  

---

## 🔗 Relevance
- **Operational planners**: optimize resource allocation and maintenance schedules.  
- **Urban planners**: improve cycling infrastructure and connectivity.  
- **Policymakers**: develop evidence-based sustainable transport strategies.  
- **Researchers**: advance academic knowledge of shared mobility systems.  
- **Public health officials**: promote active transportation to reduce pollution and improve health.  

---

## ✅ Conclusion
The analysis demonstrates how bike-sharing contributes to **sustainable urban mobility** and **public health improvements**.  
Insights from this project can guide **policy, planning, and research** in shared mobility systems.

---

## 📓 Notebook
For full code and analysis, see:  
[London Bike Sharing Analysis — August 2023 (Jupyter Notebook)](https://github.com/gekahortor/London-Bike-Sharing-Analysis/blob/2acf01a34fb7e8a24b48fd19800bb58e5bacc5f8/London_bike_sharing_analysis_august_2023_code.ipynb)

---



