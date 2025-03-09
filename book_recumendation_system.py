import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set UI Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="📚 Book Recommender", layout="wide")

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", encoding="ISO-8859-1", on_bad_lines="skip")
    df.dropna(subset=["title", "authors"], inplace=True)  # Handle missing values
    df["combined_features"] = df["title"] + " " + df["authors"]
    return df

df = load_data()

# Compute TF-IDF similarity matrix
@st.cache_data
def compute_similarity(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["combined_features"])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(df)

# Recommendation function
def recommend_books(book_title, num_recommendations=5):
    idx = df[df["title"].str.lower() == book_title.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]
    scores = list(enumerate(cosine_sim[idx]))
    sorted_books = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    return df.iloc[[i[0] for i in sorted_books]][["title", "authors", "average_rating"]]

# Top-rated books function
def top_rated_books(min_ratings=500, num_recommendations=5):
    popular_books = df[df["ratings_count"] > min_ratings].sort_values(by="average_rating", ascending=False)
    return popular_books[["title", "authors", "average_rating"]].head(num_recommendations)

# Custom UI Styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        font-family: Arial, sans-serif;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #ff5e62;
        color: white;
        font-size: 16px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("📚 AI-Powered Book Recommendation System")
st.write("Discover your next favorite book based on what you love!")

# Sidebar
st.sidebar.header("📌 Options")
num_recommendations = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# User Input
book_input = st.text_input("Enter a book title:", "")

# Recommendation Button
if st.button("🔍 Get Recommendations"):
    if book_input:
        recommendations = recommend_books(book_input, num_recommendations)
        if recommendations is None:
            st.error("❌ Book not found. Try another title!")
        else:
            st.success("✅ Here are some books you might like:")
            st.dataframe(recommendations)
    else:
        st.warning("⚠️ Please enter a book title.")

# Display Top-Rated Books
st.subheader("🌟 Top-Rated Books")
st.dataframe(top_rated_books())

st.sidebar.info("📌 Tip: Enter a book title and click the button to get recommendations!")
