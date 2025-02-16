from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
@st.cache_data
def load_data():
    file_path = "ai_use_dataset_final.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

#---------------------#
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Sample Data (Replace with actual dataset)
data = {
    "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "Willingness to Continue AI (%)": [60, 65, 70, 74, 78, 82, 85]
}
df = pd.DataFrame(data)

# Set Year as Index (for Time-Series Analysis)
df.set_index("Year", inplace=True)

# Streamlit App
st.write("### Predicting AI Adoption Trends Over Time")

st.write("""
#### **Hypothesis:**  
- AI adoption is increasing over the years.  
- We use **Time-Series Forecasting (ARIMA)** to predict future willingness to continue AI usage.
""")

st.write("Time Series Forecasting is a statistical technique used to predict future values based on previously observed data. The ARIMA (AutoRegressive Integrated Moving Average) model is one of the most commonly used methods for forecasting time-dependent data.")

# ARIMA Model for Forecasting
model = ARIMA(df["Willingness to Continue AI (%)"], order=(2, 1, 2))  # ARIMA(p, d, q)
model_fit = model.fit()
forecast = model_fit.forecast(steps=3)  # Predict for the next 3 years

# Create Future DataFrame
future_years = [2025, 2026, 2027]
future_df = pd.DataFrame({"Year": future_years, "Predicted Willingness (%)": forecast.values})

# Merge Actual & Predicted Data
df_reset = df.reset_index()
df_reset["Predicted Willingness (%)"] = None  # Fill actual data with None for plotting
full_df = pd.concat([df_reset, future_df])

# Visualization
fig = px.line(full_df, x="Year", y=["Willingness to Continue AI (%)", "Predicted Willingness (%)"],
              title="Trend of AI Adoption Over Time (With Future Predictions)",
              markers=True,
              labels={"value": "Willingness to Continue AI (%)", "variable": "Actual vs Predicted"},
              line_dash="variable")

st.plotly_chart(fig)

# Display Predictions
st.write("#### Future AI Adoption Predictions")
st.table(future_df)

st.write("""
**Interpretation:**  
- If the trend **continues upward**, AI adoption is expected to increase.  
- If the trend **stagnates or declines**, external factors (policies, AI effectiveness) might influence willingness.
""")

st.divider()

#--------------------2--------------------#

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load dataset (Replace with actual file path)
df = pd.read_csv('ai_use_dataset_final.csv')

# Define range bins with a constant difference of 50
bins = list(range(0, df['Avg Task Time (Before AI)'].max() + 50, 50))
labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

df['Time_Before_Range'] = pd.cut(df['Avg Task Time (Before AI)'], bins=bins, labels=labels, include_lowest=True)
df['Time_After_Range'] = pd.cut(df['Avg Task Time (After AI)'], bins=bins, labels=labels, include_lowest=True)

# Filter by Industry
df_industry = df.groupby('Industry')

# Compute counts and probabilities
industry_counts_before = df.groupby(['Industry', 'Time_Before_Range']).size().unstack(fill_value=0)
industry_counts_after = df.groupby(['Industry', 'Time_After_Range']).size().unstack(fill_value=0)

total_industries = df['Industry'].value_counts()

industry_prob_before = industry_counts_before.div(total_industries, axis=0)
industry_prob_after = industry_counts_after.div(total_industries, axis=0)

# Create the table
table_data = {
    "Time Range": labels,
}

for industry in df['Industry'].unique():
    table_data[f"{industry} (Before AI)"] = [industry_counts_before.loc[industry, label] if label in industry_counts_before.columns else 0 for label in labels]
    table_data[f"{industry} (After AI)"] = [industry_counts_after.loc[industry, label] if label in industry_counts_after.columns else 0 for label in labels]
    table_data[f"Probability of {industry} (Before AI)"] = [round(industry_prob_before.loc[industry, label], 4) if label in industry_prob_before.columns else 0 for label in labels]
    table_data[f"Probability of {industry} (After AI)"] = [round(industry_prob_after.loc[industry, label], 4) if label in industry_prob_after.columns else 0 for label in labels]

df_table = pd.DataFrame(table_data)

# Streamlit Web App
st.write("### AI Usage Impact Analysis by Industry")

st.write("The Naïve Bayes classifier predicts the industry based on task times before and after AI by learning patterns from historical data. Task times are grouped into 50-minute ranges, and the model calculates probabilities for each industry. When a user enters new task times, the model selects the industry with the highest probability. For example, if historical data shows that an industry typically reduces task time from 100 to 50 minutes after AI, and the user inputs the same values, the model predicts that industry as the most likely match.")

# Display Table
st.write("## Task Time Before AI vs. After AI by Industry")
st.dataframe(df_table)

# Model Training for Prediction
features = ['Avg Task Time (Before AI)', 'Avg Task Time (After AI)']
target = 'Industry'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# User Input
user_input_before = st.number_input("Enter Avg Task Time Before AI:", min_value=0, max_value=df['Avg Task Time (Before AI)'].max(), step=1)
user_input_after = st.number_input("Enter Avg Task Time After AI:", min_value=0, max_value=df['Avg Task Time (After AI)'].max(), step=1)

if user_input_before and user_input_after:
    # Predict Industry
    predicted_industry = nb_model.predict(np.array([[user_input_before, user_input_after]]))[0]
    
    # Justification
    justification = f"Based on historical data, an average task time of {user_input_before} minutes before AI and {user_input_after} minutes after AI is most commonly associated with the {predicted_industry} industry. This prediction is based on task efficiency trends and AI adoption within different industries."
    
    # Display Results
    st.write("Predicted Industry:", predicted_industry)
    st.write("Justification:", justification)