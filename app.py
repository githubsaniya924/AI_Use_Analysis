import streamlit as st

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import runpy

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("ai_use_dataset_final.csv")
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.header("Explore Data")
page = st.sidebar.radio("Go to", ["Home", "Dataset Overview", "Analysis"])

if page == "Home":
    st.markdown(
    """
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-20px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .title-container {
        text-align: left;
        color: tomato;
        font-size: 35px;
        font-weight: bold;
        animation: fadeIn 1.5s ease-in-out;
        margin-bottom: 20px;
    }
    </style>
    <div class="title-container">
        Welcome to AI Usage Analysis Dashboard: <br> 
        Explore the impact of AI usage on productivity and performance.
    </div>
    """,
    unsafe_allow_html=True
)
    runpy.run_path("home_page.py")

elif page == "Dataset Overview":
    runpy.run_path("dataset_overview.py")

elif page == "Analysis":
    st.sidebar.header("Analysis Options")
    analysis_option = st.sidebar.radio("Select Analysis Type", ["Basic", "Hypothesis Testing", "Predictive Analysis"])

    if analysis_option == "Basic":
        st.title("Basic Analysis & Insights")
        runpy.run_path("basic_analysis.py")
    
        
    elif analysis_option == "Hypothesis Testing":
        
        runpy.run_path("hypothesis_testing.py")

    elif analysis_option == "Predictive Analysis":
        
        runpy.run_path("advance_analysis.py")