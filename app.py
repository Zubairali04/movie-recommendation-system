import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Content-Based Movie Recommendation using TF-IDF")

# ------------------------------------
# Load Data
# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'overview', 'genres']]
    df.dropna(inplace=True)
    return df

movies = load_data()

# ------------------------------------
# Text Processing & Similarity
# ------------------------------------
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = compute_similarity(movies)

# ------------------------------------
# Recommendation Function
# ------------------------------------
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return []

    index = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i in similarity_scores[1:num_recommendations + 1]:
        recommended_movies.append(movies.iloc[i[0]]['title'])

    return recommended_movies

# ------------------------------------
# Streamlit UI
# ------------------------------------
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Select a Movie", movie_list)

num_recs = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie, num_recs)

    if recommendations:
        st.subheader("✅ Recommended Movies:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.warning("No recommendations found.")

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Python & Streamlit")
