<<<<<<< HEAD
# AI_Use_Analysis
analysis on ai useage dataset
Overview of the Dataset Used
The dataset used in this analysis project consists of structured data aimed at understanding AI adoption trends among students and professionals. It comprises 50,000 records and includes 25 attributes categorized into different sections. The key aspects covered in the dataset are:
Demographics & Background
User Type: Student or Working Professional
Age Group: 18-25, 26-35, 36-45, etc.
Year: The year in which the data was recorded.
Education Level: High School, Undergraduate, Postgraduate, PhD
Industry: IT, Finance, Education, Healthcare, etc.
Job Role: Intern, Manager, Developer, Full-time Student, etc.
AI Usage Patterns
AI Tools Used: ChatGPT, Grammarly, Microsoft Copilot, etc.
Frequency of AI Use: Daily, Weekly, Monthly
Purpose of AI Usage: Research, Content Creation, Data Analysis, Automation
AI Training Received: Yes, No
Efficiency & Performance Metrics
Avg Task Time (Before AI): Time in minutes before AI adoption.
Avg Task Time (After AI): Time in minutes after AI adoption.
Tasks Completed Per Week (Before AI): Number of tasks before AI use.
Tasks Completed Per Week (After AI): Number of tasks after AI use.
Work Efficiency Score: Self-rated efficiency after AI use (scale 1-10).
 AI Impact on Productivity
AI-Generated Content Usage (%): Percentage of work content generated using AI.
Perceived Increase in Productivity (%): Self-reported improvement due to AI.
Satisfaction with AI Integration: User satisfaction score (scale 1-10).
Challenges & Policies
Challenges Faced: Bias, Over-Reliance, Privacy Issues, Job Automation.
Willingness to Continue AI: Yes, No.
AI Policy Awareness: Awareness of AI policies at work/study.
Restrictions on AI Use: Fully Allowed, Limited, Banned.
Career & Academic Growth
Change in Grades/Work Performance (%): Improvement due to AI.
Job Promotions or Salary Increase: AI’s role in career growth.
Skill Development Areas: Data Analysis, Coding, Writing, etc.
The dataset was preprocessed to handle missing values, normalize numerical features, and encode categorical attributes where necessary. Statistical analysis and visualization techniques were applied to derive meaningful insights from the dataset.


Key Functionalities Implemented
This project is structured into three major sections: Basic Data Analysis, Hypothesis Testing, and Advanced Analysis. Each section applies different analytical techniques to explore AI adoption, efficiency trends, and predictive insights.
Basic Data Analysis
The first section involves exploratory data analysis (EDA) to understand AI adoption trends among users. Various data visualization techniques such as bar charts, pie charts, scatter plots, and stacked bar charts are used to provide an overview of AI impact on efficiency and productivity.
Hypothesis Testing
Hypothesis testing is a statistical method used to determine whether there is enough evidence to support a certain claim about a dataset. It helps in making data-driven decisions by testing assumptions and helps understand whether the assumption is significant or not.
The following tests are used:

Mann-Whitney U Test:

Independent Variable: Categorical (Binary or Ordinal) (e.g., "AI Training Received" → Yes/No).
Dependent Variable: Ordinal or Continuous (Scale) Data (e.g., "Work Efficiency Score" on a scale of 1-10).
  Welch ANOVA Test:
Independent Variable: Categorical (Nominal, More than 2 groups) (e.g., "Industry" → IT, Finance, Healthcare).
Dependent Variable: Continuous (Scale) Data (e.g., "Perceived Increase in Productivity (%)").
Chi-Square Test:
Both Independent & Dependent Variables are Categorical (Nominal or Ordinal).
Advanced Data Analysis
The final section incorporates machine learning techniques to predict AI adoption trends and classify efficiency levels.
Time Series Forecasting (ARIMA Model): 
Time Series Forecasting is a statistical technique used to predict future values based on previously observed data. 
The ARIMA (AutoRegressive Integrated Moving Average) model is one of the most commonly used methods for forecasting time-dependent data.
Naïve Bayes Classification:
Probabilistic Model – Based on Bayes' Theorem, it calculates the probability of a class given the input features.
Independence Assumption – Assumes that features are independent of each other within a class.
Training – Learns the probability distribution of features for each class.
Prediction – Computes probabilities for each class and selects the one with the highest probability.
Implementation Using Streamlit
The entire analysis is built using Streamlit, a Python framework for creating interactive web applications.
Interactive Visualizations using Plotly, Seaborn, and Matplotlib - python Libraries.
Deployment on Streamlit Community
The project is deployed on Streamlit Community Cloud, making it accessible without requiring local installation

