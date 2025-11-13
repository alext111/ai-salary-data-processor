import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
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

# Histogram of salary distribution
sns.histplot(dataset['salary_usd'], bins=30, kde=True)
plt.title("Salary Distribution (USD)")
plt.xlabel("Salary (USD)")
plt.ylabel("Frequency")
plt.show()

# Box plot of salary sorted by experience level
sns.boxplot(x='experience_level', y='salary_usd', data=dataset, order=["EN","MI","SE","EX"])
plt.title("Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Salary (USD)")
plt.show()

# Top 10 locations with highest average salary
avg_salary_location = dataset.groupby('company_location')['salary_usd'].mean().sort_values(ascending=False).head(15)

sns.barplot(x=avg_salary_location.values, y=avg_salary_location.index)
plt.title("Top 10 Locations by Average Salary")
plt.xlabel("Average Salary (USD)")
plt.ylabel("Location")
plt.show()

# Box plot of salary based on highest education level
sns.boxplot(x='education_required', y='salary_usd', data=dataset, order=["Associate","Bachelor","Master","PhD"])
plt.title("Salary by Education Level")
plt.xlabel("Education Required")
plt.ylabel("Salary (USD)")
plt.show()

# Top 10 industries with highest average salary
avg_salary_industry = dataset.groupby('industry')['salary_usd'].mean().sort_values(ascending=False).head(10)

sns.barplot(x=avg_salary_industry.values, y=avg_salary_industry.index)
plt.title("Top 10 Industries by Average Salary")
plt.xlabel("Average Salary (USD)")
plt.ylabel("Industry")
plt.show()

# Scatter plot of salaries versus years of experience
sns.scatterplot(x='years_experience', y='salary_usd', hue='experience_level', data=dataset)
plt.title("Salary vs. Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (USD)")
plt.show()

'''
# Sidebar filters
selected_location = st.sidebar.selectbox("Select Country", sorted(dataset['company_location'].unique()))
selected_industry = st.sidebar.selectbox("Select Industry", sorted(dataset['industry'].unique()))
selected_experience = st.sidebar.multiselect("Experience Level", dataset['experience_level'].unique())

filtered = dataset[
    (dataset['company_location'] == selected_location) &
    (dataset['industry'] == selected_industry)
]
if selected_experience:
    filtered = filtered[filtered['experience_level'].isin(selected_experience)]

st.subheader("Salary Distribution (USD)")
fig, ax = plt.subplots()
sns.histplot(filtered['salary_usd'], bins=30, kde=True, ax=ax)
st.pyplot(fig)
'''