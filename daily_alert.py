import pandas as pd
import joblib

def filter_jobs_for_alerts(df, allowed_clusters=None, keyword_filters=None):
    """
    Filter jobs based on cluster IDs or keywords in title or skills.
    - allowed_clusters: list of int cluster IDs to alert on
    - keyword_filters: list of keywords to match in title or skills (case-insensitive)
    """
    if allowed_clusters is not None:
        df = df[df['cluster'].isin(allowed_clusters)]
    if keyword_filters is not None:
        keywords_pattern = '|'.join(keyword_filters)
        mask = df['title'].str.contains(keywords_pattern, case=False, na=False) | \
               df['skills'].str.contains(keywords_pattern, case=False, na=False)
        df = df[mask]
    return df

def main():
    print("Loading clustered jobs data...")
    df_jobs = pd.read_csv('clustered_jobs.csv')

    # Set your filters here:
    allowed_clusters = [0, 1, 2]       # example clusters to alert on
    keyword_filters = ['python', 'data science', 'machine learning']  # example keywords

    filtered_jobs = filter_jobs_for_alerts(df_jobs, allowed_clusters, keyword_filters)
    filtered_jobs = filtered_jobs.drop_duplicates(subset=['title', 'company'])


    if filtered_jobs.empty:
        print("No new matching jobs found for alert.")
    else:
        print(f"Found {len(filtered_jobs)} matching jobs:")
        for idx, job in filtered_jobs.iterrows():
            print(f"title: {job['title']}")
            print(f"company: {job['company']}")
            print(f"cluster: {job['cluster']}")
            print(f"skills: {job['skills']}")
            print("-" * 40)

if __name__ == "__main__":
    main()
