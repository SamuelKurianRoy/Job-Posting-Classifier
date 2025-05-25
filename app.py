import streamlit as st
import pandas as pd
import joblib

# Load preprocessed clustered job data
@st.cache_data
def load_jobs():
    return pd.read_csv("clustered_jobs.csv")

# Filter jobs by clusters and keywords
def filter_jobs(df, clusters, keywords):
    keywords_pattern = '|'.join(keywords)
    mask = df['title'].str.contains(keywords_pattern, case=False, na=False) | \
           df['skills'].str.contains(keywords_pattern, case=False, na=False)
    return df[df['cluster'].isin(clusters) & mask]

# Streamlit app UI
st.title("Daily Job Alerts - Streamlit")

df_jobs = load_jobs()

# Sidebar filters
st.sidebar.header("Filter Options")
selected_clusters = st.sidebar.multiselect("Select Clusters", sorted(df_jobs['cluster'].unique()), default=[0, 1, 2])
keywords_input = st.sidebar.text_input("Enter Keywords (comma-separated)", value="python,data,AI")

if st.sidebar.button("Filter Jobs"):
    keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
    filtered = filter_jobs(df_jobs, selected_clusters, keywords)
    st.write(f"### {len(filtered)} jobs matched your criteria:")
    st.dataframe(filtered[['title', 'company', 'skills', 'cluster']])
else:
    st.write("Use the sidebar to filter jobs.")
