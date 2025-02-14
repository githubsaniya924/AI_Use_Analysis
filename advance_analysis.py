import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    file_path = "ai_use_dataset_final.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Title
st.title("📊 AI Usage: Clustering & Prediction Analysis")

st.write("""
### 1️⃣ Data Preprocessing & Encoding
""")

# Display raw data
st.write("🔹 **Sample Dataset:**", df.head())

# Handle categorical variables
st.write("🔹 **Encoding Categorical Data...**")

# Mapping Frequency of AI Use (Ordinal Encoding)
freq_mapping = {'Daily': 3, 'Weekly': 2, 'Monthly': 1}
if 'Frequency of AI Use' in df.columns:
    df['Frequency of AI Use'] = df['Frequency of AI Use'].map(freq_mapping)

# Drop NaN values after encoding
df = df.dropna(subset=['Frequency of AI Use', 'Perceived Increase in Productivity (%)'])

st.write("✅ **Data after Encoding:**", df.head())

# --- Clustering (K-Means) ---
st.write("""
### 2️⃣ Clustering Analysis (K-Means)
#### Grouping users based on AI Usage Frequency & Productivity Increase
""")

# Select features for clustering
clustering_data = df[['Frequency of AI Use', 'Perceived Increase in Productivity (%)']]

# K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clustering_data['Cluster'] = kmeans.fit_predict(clustering_data)

# Plot Clusters
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=clustering_data['Frequency of AI Use'], 
                y=clustering_data['Perceived Increase in Productivity (%)'], 
                hue=clustering_data['Cluster'], palette="Set1", ax=ax)
ax.set_title("K-Means Clustering of AI Usage & Productivity Increase")
ax.set_xlabel("Frequency of AI Use (Encoded)")
ax.set_ylabel("Perceived Increase in Productivity (%)")
st.pyplot(fig)

# --- Prediction (Linear Regression) ---
st.write("""
### 3️⃣ Prediction Analysis (Linear Regression)
#### Can Frequency of AI Use Predict Perceived Productivity Increase?
""")

# Prepare data for prediction
X = df[['Frequency of AI Use']]
y = df['Perceived Increase in Productivity (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"🔹 **R² Score:** {r2:.2f}")

# Plot Regression Line
fig, ax = plt.subplots(figsize=(6, 4))
sns.regplot(x=X_test, y=y_pred, scatter_kws={"color": "blue"}, line_kws={"color": "red"}, ax=ax)
ax.set_title("Linear Regression: AI Usage vs Productivity")
ax.set_xlabel("Frequency of AI Use (Encoded)")
ax.set_ylabel("Predicted Productivity Increase (%)")
st.pyplot(fig)

st.success("✅ **Analysis Completed!**")
