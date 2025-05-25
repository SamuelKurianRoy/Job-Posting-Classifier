# train_model.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# --- 1. Scrape job postings ---
def scrape_jobs(keyword="data science", pages=3):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping page: {page}")
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}, status: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "title": title,
                    "company": company,
                    "skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue

        time.sleep(1)

    return pd.DataFrame(jobs_list)


# --- 2. Preprocess skills ---
def preprocess_skills(df):
    df['skills_clean'] = df['skills'].str.lower().str.strip()
    return df


# --- 3. Vectorize skills ---
def custom_tokenizer(text):
    return text.split(',')

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=True)
    X = vectorizer.fit_transform(df['skills_clean'])
    return X, vectorizer



# --- 4. Apply clustering methods ---
def cluster_and_evaluate(X):
    models = {
        'KMeans': KMeans(n_clusters=5, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=2),
        'Agglomerative': AgglomerativeClustering(n_clusters=5)
    }

    best_model_name = None
    best_model = None
    best_score = -1
    best_labels = None

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            if name == 'DBSCAN':
                labels = model.fit_predict(X.toarray())
            else:
                labels = model.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= 1:
                print(f"Skipping {name}: only {n_clusters} cluster(s) found")
                continue

            score = silhouette_score(X, labels)
            print(f"{name} Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model_name = name
                best_model = model
                best_labels = labels
        except Exception as e:
            print(f"Error with {name}: {e}")

    print(f"\nBest model: {best_model_name} with Silhouette Score: {best_score:.4f}")
    return best_model_name, best_model, best_labels


def main():
    print("Starting job scraping...")
    df = scrape_jobs()
    if df.empty:
        print("No jobs found. Exiting.")
        return
    print(f"Scraped {len(df)} jobs.")

    print("Preprocessing skills...")
    df = preprocess_skills(df)

    print("Vectorizing skills...")
    X, vectorizer = vectorize_skills(df)

    print("Clustering and evaluating models...")
    best_model_name, best_model, best_labels = cluster_and_evaluate(X)

    print("Saving model and vectorizer...")
    joblib.dump(best_model, 'best_clustering_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

    df['cluster'] = best_labels
    df.to_csv('clustered_jobs.csv', index=False)

    print("Training complete. Files saved:")
    print("- best_clustering_model.joblib")
    print("- tfidf_vectorizer.joblib")
    print("- clustered_jobs.csv")


if __name__ == "__main__":
    main()