import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# ------------------------------------
# Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------------
# Custom CSS Styling
# ------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: #ffffff;
        min-height: 100vh;
    }
    
    .main-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
        text-align: center;
        padding: 20px 0;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .movie-card {
        background: #f8f9fa;
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .genre-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 25px;
        font-size: 12px;
        font-weight: 500;
        margin: 3px;
        text-transform: capitalize;
    }
    
    .genre-action { background: #667eea; color: white; }
    .genre-adventure { background: #f5576c; color: white; }
    .genre-animation { background: #4facfe; color: white; }
    .genre-comedy { background: #43e97b; color: #1a1a2e; }
    .genre-crime { background: #434343; color: white; }
    .genre-documentary { background: #a8edea; color: #1a1a2e; }
    .genre-drama { background: #ff9a9e; color: #1a1a2e; }
    .genre-family { background: #fbc2eb; color: #1a1a2e; }
    .genre-fantasy { background: #fa709a; color: white; }
    .genre-history { background: #d299c2; color: #1a1a2e; }
    .genre-horror { background: #2c3e50; color: white; }
    .genre-music { background: #ffecd2; color: #1a1a2e; }
    .genre-mystery { background: #3a1c71; color: white; }
    .genre-romance { background: #ff758c; color: white; }
    .genre-scifi { background: #00c6ff; color: white; }
    .genre-thriller { background: #1e3c72; color: white; }
    .genre-war { background: #c21500; color: white; }
    .genre-western { background: #c79081; color: #1a1a2e; }
    .genre-default { background: #636fa4; color: white; }
    
    [data-testid="stSidebar"] {
        background: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .movie-details {
        background: #e9ecef;
        border-radius: 15px;
        padding: 15px;
        margin-top: 10px;
    }
    
    .movie-overview {
        color: #495057;
        line-height: 1.6;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    .footer {
        text-align: center;
        padding: 30px;
        color: #6c757d;
        font-size: 14px;
        border-top: 1px solid #e9ecef;
        margin-top: 50px;
    }
    
    .similar-link {
        color: #0072ff;
        text-decoration: none;
        padding: 5px 10px;
        border-radius: 10px;
        background: #e9ecef;
        transition: all 0.3s ease;
        display: inline-block;
        margin: 3px;
        cursor: pointer;
    }
    
    .similar-link:hover {
        background: #dee2e6;
    }
    
    /* Text colors for white background */
    h1, h2, h3, h4, h5, h6, p, div {
        color: #1a1a2e;
    }
    
    .stMarkdown {
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# Genre Color Mapping
# ------------------------------------
GENRE_COLORS = {
    'action': 'genre-action',
    'adventure': 'genre-adventure',
    'animation': 'genre-animation',
    'comedy': 'genre-comedy',
    'crime': 'genre-crime',
    'documentary': 'genre-documentary',
    'drama': 'genre-drama',
    'family': 'genre-family',
    'fantasy': 'genre-fantasy',
    'history': 'genre-history',
    'horror': 'genre-horror',
    'music': 'genre-music',
    'mystery': 'genre-mystery',
    'romance': 'genre-romance',
    'sci-fi': 'genre-scifi',
    'thriller': 'genre-thriller',
    'war': 'genre-war',
    'western': 'genre-western'
}

def get_genre_badge_html(genre):
    genre_lower = genre.lower().strip()
    css_class = GENRE_COLORS.get(genre_lower, 'genre-default')
    return f'<span class="genre-badge {css_class}">{genre}</span>'

def get_all_genres_html(genres_str):
    genres = genres_str.split('|')
    return ''.join([get_genre_badge_html(g.strip()) for g in genres])

def get_movie_poster(title):
    hash_val = int(hashlib.md5(title.encode()).hexdigest(), 16)
    poster_id = (hash_val % 1000) + 1
    return f"https://picsum.photos/seed/{poster_id}/300/450"

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
# Recommendation Function
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

        if genre_filter and genre_filter != "All":
            if genre_filter not in movie['genres']:
                continue

        recommendations.append(movie)

        if len(recommendations) == top_n:
            break

    return recommendations

# ------------------------------------
# Get Similar Movies
# ------------------------------------
def get_similar_movies(movie_title, count=3):
    if movie_title not in movies['title'].values:
        return []
    
    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similar = []
    for i in similarity_scores[1:count+1]:
        similar.append(movies.iloc[i[0]]['title'])
    return similar

# ------------------------------------
# Sidebar
# ------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Settings")
    
    st.markdown("### 📊 Statistics")
    st.markdown(f"""
    <div class="stats-card">
        <h3 style="margin:0; color: white;">{len(movies)}</h3>
        <p style="margin:0; color: rgba(255,255,255,0.8);">Movies Available</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-card">
        <h3 style="margin:0; color: white;">{len(all_genres)}</h3>
        <p style="margin:0; color: rgba(255,255,255,0.8);">Unique Genres</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Filter Options")
    
    selected_genre_filter = st.selectbox(
        "🎭 Filter by Genre",
        ["All"] + all_genres
    )
    
    st.markdown("---")
    st.markdown("### ⚡ Display Options")
    col1, col2 = st.columns(2)
    with col1:
        show_posters = st.checkbox("Posters", value=True)
    with col2:
        show_details = st.checkbox("Details", value=True)

# ------------------------------------
# Main Content
# ------------------------------------
st.markdown('<h1 class="main-title">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6c757d; font-size: 1.2rem;">✨ Enhanced Content-Based Recommendation Engine ✨</p>', unsafe_allow_html=True)

st.markdown("### 🔎 Find Your Next Favorite Movie")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "🎥 Select a Movie",
        movie_list,
        help="Search and select a movie you like"
    )

with col2:
    filter_genre = st.selectbox(
        "🎭 Quick Genre Filter",
        ["All Genres"] + all_genres,
        help="Filter movies by genre"
    )

with col3:
    num_recs = st.number_input(
        "📌 Number",
        min_value=1,
        max_value=10,
        value=5
    )

# ------------------------------------
# Show selected movie info
# ------------------------------------
if selected_movie:
    selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
    
    st.markdown("---")
    st.markdown(f"### 📽️ Selected: **{selected_movie}**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if show_posters:
            poster_url = get_movie_poster(selected_movie)
            st.image(poster_url, caption=selected_movie, use_container_width=True)
    
    with col2:
        st.markdown(f"**🎭 Genres:**")
        st.markdown(get_all_genres_html(selected_movie_data['genres']), unsafe_allow_html=True)
        
        st.markdown(f"**📝 Overview:**")
        st.markdown(f'<p class="movie-overview">{selected_movie_data["overview"]}</p>', unsafe_allow_html=True)
        
        similar_movies = get_similar_movies(selected_movie, 5)
        if similar_movies:
            st.markdown("**🔗 Similar Movies:**")
            for m in similar_movies:
                st.markdown(f'<span class="similar-link">{m}</span>', unsafe_allow_html=True)

# ------------------------------------
# Recommendation Button
# ------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    recommend_btn = st.button("✨ Get Recommendations", use_container_width=True)

# ------------------------------------
# Display Recommendations
# ------------------------------------
if recommend_btn:
    with st.spinner('🎬 Analyzing movie features and finding similar movies...'):
        import time
        time.sleep(1)
        results = recommend_movies(selected_movie, selected_genre_filter, num_recs)
    
    if results:
        st.markdown("---")
        st.markdown(f"### ✅ Recommended Movies for You")
        
        for idx, movie in enumerate(results):
            with st.container():
                st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 3]) if show_posters else [1]
                
                with col1:
                    if show_posters:
                        poster_url = get_movie_poster(movie['title'])
                        st.image(poster_url, caption=f"#{idx+1}", use_container_width=True, output_format="auto")
                
                with col2:
                    st.markdown(f"#### 🎬 {idx+1}. {movie['title']}")
                    st.markdown(get_all_genres_html(movie['genres']), unsafe_allow_html=True)
                    st.markdown(f"**📝 Overview:**")
                    overview_text = movie['overview'][:300] + '...' if len(movie['overview']) > 300 else movie['overview']
                    st.markdown(f'<p class="movie-overview">{overview_text}</p>', unsafe_allow_html=True)
                    
                    if show_details:
                        with st.expander("📋 More Details"):
                            st.markdown(f"""
                            <div class="movie-details">
                                <p><strong>🎭 Genres:</strong> {movie['genres']}</p>
                                <p><strong>📝 Full Overview:</strong> {movie['overview']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    similar_to_this = get_similar_movies(movie['title'], 3)
                    if similar_to_this:
                        st.markdown("**You might also like:**")
                        links = ', '.join([f'`{m}`' for m in similar_to_this])
                        st.markdown(links)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.warning("No movies found matching your criteria. Try adjusting your filters!")

# ------------------------------------
# Browse by Genre
# ------------------------------------
st.markdown("---")
st.markdown("### 🎭 Browse Movies by Genre")

genre_tabs = st.tabs(all_genres[:8])

for i, genre in enumerate(all_genres[:8]):
    with genre_tabs[i]:
        genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
        st.markdown(f"**{genre}** - {len(genre_movies)} movies found")
        
        for j, (_, movie) in enumerate(genre_movies.head(6).iterrows()):
            with st.expander(f"🎬 {movie['title']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    if show_posters:
                        poster_url = get_movie_poster(movie['title'])
                        st.image(poster_url, use_container_width=True)
                with col2:
                    st.markdown(get_all_genres_html(movie['genres']), unsafe_allow_html=True)
                    st.markdown(f"_{movie['overview'][:200]}..._")

# ------------------------------------
# Footer
# ------------------------------------
st.markdown("""
<div class="footer">
    <p>🎬 Movie Recommendation System | Built with Streamlit & TF-IDF</p>
    <p>📚 Chicago State University | Md. Zubair Ali</p>
    <p>Made with ❤️ using Machine Learning</p>
</div>
""", unsafe_allow_html=True)
