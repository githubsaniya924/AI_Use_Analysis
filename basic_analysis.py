import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

@st.cache_data
def load_data():
    df = pd.read_csv("ai_use_dataset_final.csv")
    return df

df = load_data()

    
# Filters
st.sidebar.header("Filters")
User_Type = st.sidebar.selectbox("Select User_Type", ["All"] + df["User_Type"].unique().tolist())
if User_Type != "All":
    df = df[df["User_Type"] == User_Type]


fig2 = px.bar(
    df, 
    x="User_Type", 
    title="User_Type Distribution", 
    color="User_Type",  # Assigns different colors to bars
    color_discrete_sequence=px.colors.qualitative.Set2  # Attractive color scheme
)

fig2.update_layout(bargap=0.3)  # Reduces bar width
st.plotly_chart(fig2)

st.markdown("""
The User_Type Distribution output is a bar chart showing the count of each user type in the dataset.  
In the above bar chart, the number of students is greater than the number of employees using AI.
""")


if "Avg Task Time (Before AI)" in df.columns and "Avg Task Time (After AI)" in df.columns:
    fig3 = px.bar(
        df, 
        x="User_Type", 
        y=["Avg Task Time (Before AI)", "Avg Task Time (After AI)"], 
        barmode="group", 
        title="Task Time Before vs After AI",
        color_discrete_sequence=px.colors.qualitative.Pastel  # Using an attractive color scheme
    )

    st.plotly_chart(fig3)

    st.markdown(""" 
    The Task Time Before vs After AI output is a grouped bar chart comparing the average task completion times 
    for each user type before and after AI usage.  
    It is observed that the time taken to complete a task after AI has dropped for both students and employees.
    """)

# Skill Development Areas     
if "Skill Development Areas" in df.columns:
    skill_counts = df["Skill Development Areas"].value_counts().reset_index()
    skill_counts.columns = ["Skill", "Count"]
    fig4 = px.pie(skill_counts, names="Skill", values="Count", title="Skill Development Areas")
    st.plotly_chart(fig4)
st.markdown(""" The Skill Development Areas output is a pie chart representing the distribution of different skill areas developed through AI usage. 
            It highlights which skills users have focused on the most, providing insights into the key areas of growth.""")



# Filter challenges separately for employees and students
df_challenges_employee = df[df["User_Type"] == "Employee"]["Challenges Faced"].value_counts().reset_index()
df_challenges_student = df[df["User_Type"] == "Student"]["Challenges Faced"].value_counts().reset_index()

# Rename columns for clarity
df_challenges_employee.columns = ["Challenge", "Count"]
df_challenges_employee["User_Type"] = "Employee"

df_challenges_student.columns = ["Challenge", "Count"]
df_challenges_student["User_Type"] = "Student"

# Combine both datasets
df_challenges_combined = pd.concat([df_challenges_employee, df_challenges_student])

# Create the grouped horizontal bar chart
fig = px.bar(df_challenges_combined, 
            x="Count", 
            y="Challenge", 
            color="User_Type", 
            barmode="group",  # Group bars for comparison
            title="Major Challenges in AI Adoption (Employees vs. Students)",
            labels={"Challenge": "Challenges", "Count": "Frequency"},
            orientation="h",  # Horizontal bar chart
            color_discrete_map={"Employee": "#636EFA", "Student": "#EF553B"}  # Custom colors
)

# Improve layout aesthetics
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
    font=dict(size=14, color="black"),  # Improve font size and color
    margin=dict(l=100, r=50, t=50, b=50),  # Adjust margins
    xaxis_title="Frequency",
    yaxis_title="Challenges",
    legend_title="User_Type"
)

# Display in Streamlit
st.plotly_chart(fig)
st.markdown("""
The Major Challenges in AI Adoption (Employees vs. Students) output is a grouped horizontal bar chart comparing the most common challenges faced by employees and students in AI adoption. 
            It highlights differences in concerns and obstacles, providing insights into how AI adoption experiences vary across user types. 
""")

#highest usage of AI tools?
df_ai_tools = df["AI Tools Used"].value_counts().reset_index()

# Rename columns for clarity
df_ai_tools.columns = ["AI Tools Used", "Usage Count"]

# Create a donut chart
# Soft and diverse colors
fig = px.pie(df_ai_tools, 
        names="AI Tools Used",  # Correct column name
        values="Usage Count", 
        title="AI Tool Usage Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3)

# Customize layout for better visibility
fig.update_layout(
    font=dict(size=14, color="black"),  # Better font size and color
    legend_title="AI Tools",
    showlegend=True
)

st.plotly_chart(fig)
st.markdown("""
The AI Tool Usage Distribution output is a donut chart displaying the most frequently used AI tools. 
It visually represents the proportion of different AI tools utilized by users, highlighting the most popular ones based on usage count. 
""")



# Example dataset (Age vs. AI Usage Count for multiple years)
df_example = pd.DataFrame({
    "Age": ["18-25", "26-35", "36-45", "46-55", "56-65"] * 3,
    "AI Users Count": [150, 200, 180, 130, 100, 220, 270, 260, 190, 140, 300, 320, 280, 220, 160],
    "Year": ["2020"] * 5 + ["2022"] * 5 + ["2024"] * 5  # Three categories like the image
})

# Create scatter plot
fig = px.scatter(df_example, 
                x="Age", 
                y="AI Users Count", 
                color="Year",  # Different colors for each year
                size="AI Users Count",  # Bigger dots for more users
                title="AI Usage by Age Over Years",
                labels={"Age": "Age", "AI Users Count": "AI Users"},
                hover_name="Age",
                opacity=0.9  # Slight transparency for better visibility
)

# Improve layout and hover effect
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",  
    paper_bgcolor="rgba(0,0,0,0)",  
    font=dict(size=14, color="black"),
    margin=dict(l=80, r=50, t=50, b=50),
    xaxis_title="Age",
    yaxis_title="Number of AI Users",
    hovermode="x unified"  # Ensures smooth hover effect
)

# Display in Streamlit
st.plotly_chart(fig)
st.markdown("""The AI Usage by Age Over Years output is a scatter plot showing how AI adoption varies across different age 
        groups over multiple years. Each point represents the number of AI users in a specific age range, with larger dots 
            indicating higher usage. The color differentiation highlights trends in AI adoption over time. """)



# Count occurrences of each category
awareness_counts = df.groupby(["User_Type", "Company/University AI Policy Awareness"]).size().unstack()

# Create a Stacked Bar Chart
st.markdown("""**AI Policy Awareness: Students vs. Employees**""")

fig, ax = plt.subplots(figsize=(8, 5))
awareness_counts.plot(kind="bar", stacked=True, ax=ax, color=["#6a0dad", "#dda0dd"])  # Purple shades

# Customize chart
ax.set_xlabel("User Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Stacked Bar Chart of AI Policy Awareness", fontsize=14)
ax.legend(title="AI Policy Awareness", labels=["No", "Yes"])
plt.xticks(rotation=0)  # Keep labels horizontal

# Show plot in Streamlit
st.pyplot(fig)
st.markdown("""This stacked bar chart visualizes the awareness of AI policies among students and employees. The X-axis represents the user type (Students vs. Employees), while the Y-axis shows the count of responses.""")