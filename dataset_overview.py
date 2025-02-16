import streamlit as st
import pandas as pd

# Custom Styling & Animations
st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        h1 { color: tomato; text-align: center; background-color: white; }
        .section-title {
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .description {
            text-align: center;
            font-size: 18px;
            line-height: 1.6;
            color: #444;
            margin-bottom: 20px;
        }
        .stMarkdown { font-size: 18px; }
        .block-container { padding: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Introduction
st.title("AI Usage Dataset Overview")
st.markdown(
    '<p class="description">This dataset contains 50,000 records analyzing AI adoption trends among students and professionals. It covers demographics, AI usage patterns, efficiency changes, and challenges faced. The dataset includes 25 attributes, categorized into different sections for clarity.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# First 5 Rows of the dataset
df = pd.read_csv("ai_use_dataset_final.csv")
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">First 5 Rows of the dataset</h5>', unsafe_allow_html=True)
st.dataframe(df.head())

st.markdown("---")

# Dataset Attributes & Explanation
st.markdown('<h4>Dataset Attributes & Explanation</h4>',unsafe_allow_html=True)

# Demographics & Background
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">Demographics & Background</h5>', unsafe_allow_html=True)
st.markdown("""
- **User Type**: Student or Working Professional  
- **Age Group**: 18-25, 26-35, etc.  
- **Year**: Year of data recording  
- **Education Level**: High School, Undergraduate, Postgraduate, PhD  
- **Industry**: IT, Finance, Education, Healthcare, etc.  
- **Job Role**: Intern, Manager, Developer, Student, etc.  
""")

# AI Usage Patterns
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">AI Usage h5atterns</h5>', unsafe_allow_html=True)
st.markdown("""
- **AI Tools Used**: ChatGPT, Grammarly, Microsoft Copilot, etc.  
- **Frequency of AI Use**: Daily, Weekly, Monthly  
- **Purpose of AI Usage**: Research, Content Creation, Data Analysis, Automation  
- **AI Training Received**: Yes, No  
""")

# Efficiency & Performance Metrics
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">Efficiency & Performance Metrics</h5>', unsafe_allow_html=True)
st.markdown("""
- **Avg Task Time (Before AI)**: Time (minutes) before AI  
- **Avg Task Time (After AI)**: Time (minutes) after AI  
- **Tasks Completed Per Week (Before AI)**: Weekly tasks before AI  
- **Tasks Completed Per Week (After AI)**: Weekly tasks after AI  
- **Work Efficiency Score**: Self-rated (1-10)  
""")

# AI Impact on Productivity
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">AI Impact on Productivity</h5>', unsafe_allow_html=True)
st.markdown("""
- **AI-Generated Content Usage (%)**: Percentage of AI-generated work  
- **Perceived Increase in Productivity (%)**: Productivity increase due to AI  
- **Satisfaction with AI Integration**: User satisfaction (1-10)  
""")

# Challenges & Policies
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">Challenges & Policies</h5>', unsafe_allow_html=True)
st.markdown("""
- **Challenges Faced**: Bias, Over-Reliance, Privacy Issues, Job Automation  
- **Willingness to Continue AI**: Yes, No  
- **AI Policy Awareness**: Yes, No  
- **Restrictions on AI Use**: Fully Allowed, Limited, Banned  
""")

# Career & Academic Growth
st.markdown('<h5 class="section-title" style="background-color:white; color:tomato;">Career & Academic Growth</h5>', unsafe_allow_html=True)
st.markdown("""
- **Change in Grades/Work Performance (%)**: Performance improvement due to AI  
- **Job Promotions or Salary Increase**: Career impact (Yes, No)  
- **Skill Development Areas**: AI-assisted skill enhancement  
""")

# Footer
st.markdown("---")
st.info("This dataset provides valuable insights into AI adoption trends, productivity shifts, and ethical concerns in AI usage.")


