import re
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')



stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and preprocess data
data = pd.read_csv("D:/flask/job_data.csv")
data['description'] = data['description'].apply(preprocess_text)
job_descriptions = data['description'].tolist()
job_data = data.values.tolist()

word2vec_model = Word2Vec(sentences=job_descriptions, vector_size=100, window=5, min_count=1, workers=4)
def get_average_word2vec(tokens, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    for token in tokens:
        if token in model.wv:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[token])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

num_features = 100  # Should match the vector_size parameter of Word2Vec
job_descriptions_vec = np.array([get_average_word2vec(desc, word2vec_model, num_features) for desc in job_descriptions])

k = 7

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans_labels = kmeans.fit_predict(job_descriptions_vec)
kmeans_silhouette = silhouette_score(job_descriptions_vec, kmeans_labels)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(job_descriptions_vec)
dbscan_silhouette = silhouette_score(job_descriptions_vec, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1





# Compare silhouette scores
print(f"KMeans Silhouette Score: {kmeans_silhouette}")
print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")


# Function to recommend jobs based on user skills
def recommend_jobs(user_skills, algorithm='kmeans'):
    user_skills_tokens = preprocess_text(' '.join(user_skills))
    user_skills_vec = get_average_word2vec(user_skills_tokens, word2vec_model, num_features).reshape(1, -1)

    if algorithm == 'kmeans':
        cluster_distances = kmeans.transform(user_skills_vec)
        cluster_index = np.argmin(cluster_distances)
        cluster_labels = kmeans_labels
    elif algorithm == 'dbscan':
        closest, _ = pairwise_distances_argmin_min(user_skills_vec, job_descriptions_vec)
        cluster_index = dbscan_labels[closest[0]]
        cluster_labels = dbscan_labels

    else:
        raise ValueError("Unknown algorithm: choose from 'kmeans', 'dbscan'")

    cluster_job_indices = np.where(cluster_labels == cluster_index)[0]
    similarities = cosine_similarity(user_skills_vec, job_descriptions_vec[cluster_job_indices])
    top_similar_indices = similarities.argsort()[0][-10:][::-1]
    recommended_jobs = [job_data[cluster_job_indices[i]] for i in top_similar_indices]

    return recommended_jobs

user_skills = ['python', 'machine learning', 'data analysis']
for algo in ['kmeans', 'dbscan']:
    print(f"\nRecommendations using {algo}:")
    recommendations = recommend_jobs(user_skills, algorithm=algo)
    for job in recommendations:
        print(job)


