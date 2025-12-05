import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Kaggle dataset for AI Job Market & Salary Analysis 2025
# https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025/data

dataset = pd.read_csv("ai_job_dataset.csv")

dataset.info()
dataset.isnull().sum()
dataset.head()

# Relevant features
dataset = dataset[['job_title', 'salary_usd', 'experience_level', 'company_location', 'education_required', 'years_experience', 'industry', 'company_name']]

# Streamlit
# Sidebar filters
selected_location = st.sidebar.selectbox("Select Country", sorted(dataset['company_location'].unique()))
selected_experience = st.sidebar.multiselect("Experience Level", dataset['experience_level'].unique())

filtered = dataset[
    (dataset['company_location'] == selected_location)
]
if selected_experience:
    filtered = filtered[filtered['experience_level'].isin(selected_experience)]

st.subheader("Salary Distribution (USD)")
fig, ax = plt.subplots()
sns.histplot(filtered['salary_usd'], bins=30, kde=True, ax=ax)
st.pyplot(fig)