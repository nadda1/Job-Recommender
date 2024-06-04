import re 
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"  


app = Flask(__name__)

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



data  = pd.read_csv("D:/flask/job_data.csv")
job_descriptions = [description for description in data['description']]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
job_descriptions_tfidf = tfidf_vectorizer.fit_transform(job_descriptions)
job_data = data.values.tolist()


k = 5


kmeans = KMeans(n_clusters=k, random_state=42)
cluster_assignments = kmeans.fit_predict(job_descriptions_tfidf)

# Define API endpoint
@app.route('/recommend', methods=['Get','POST'])
def recommend_jobs():
    try:
        user_skills = request.json['skills']
        user_skills_str = ' '.join(user_skills)
        user_skills_tfidf = tfidf_vectorizer.transform([user_skills_str])
        print(user_skills)
        print(user_skills_str)

        user_cluster_distances = kmeans.transform(user_skills_tfidf)
        user_cluster_index = np.argmin(user_cluster_distances)

        
        cluster_job_indices = np.where(cluster_assignments == user_cluster_index)[0]

        similarities = cosine_similarity(user_skills_tfidf, job_descriptions_tfidf[cluster_job_indices])

        top_similar_indices = similarities.argsort()[0][-5:][::-1]

        recommended_jobs = [job_data[cluster_job_indices[i]] for i in top_similar_indices]

        return jsonify({'recommended_jobs': recommended_jobs})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
