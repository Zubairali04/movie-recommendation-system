import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Enhanced Content-Based Recommendation System")

# ------------------------------------
# Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['title', 'overview', 'genres']]
    df.dropna(inplace=True)
    return df

movies = load_data()

# ------------------------------------
# TF-IDF & Similarity
# ------------------------------------
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = compute_similarity(movies)

# ------------------------------------
# Extract Unique Genres
# ------------------------------------
def get_all_genres():
    genres = set()
    for g in movies['genres']:
        for genre in g.split('|'):
            genres.add(genre)
    return sorted(genres)

all_genres = get_all_genres()

# ------------------------------------
# Recommendation Function (ENHANCED)
# ------------------------------------
def recommend_movies(movie_title, genre_filter, top_n=5):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in similarity_scores[1:]:
        movie = movies.iloc[i[0]]

        # Genre filter logic
        if genre_filter != "All":
            if genre_filter not in movie['genres']:
                continue

        recommendations.append(movie)

        if len(recommendations) == top_n:
            break

    return recommendations

# ------------------------------------
# Streamlit UI
# ------------------------------------
movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Select a Movie", movie_list)

selected_genre = st.selectbox(
    "🎭 Filter by Genre",
    ["All"] + all_genres
)

num_recs = st.slider("📌 Number of Recommendations", 1, 5, 3)

if st.button("Recommend Movies"):
    results = recommend_movies(selected_movie, selected_genre, num_recs)

    if results:
        st.subheader("✅ Recommended Movies")
        for movie in results:
            st.markdown(f"### 🎬 {movie['title']}")
            st.write(f"📝 **Overview:** {movie['overview']}")
            st.write(f"🎭 **Genres:** {movie['genres']}")
            st.markdown("---")
    else:
        st.warning("No movies found for the selected genre.")

# ------------------------------------
# Footer
# ------------------------------------
st.caption("University Project | Python • ML • Streamlit")
