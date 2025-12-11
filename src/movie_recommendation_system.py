# src/movie_recommendation_system.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast

# -----------------------------
# Load the dataset
# -----------------------------
movies = pd.read_csv("data/movies.csv")  # Only use movies.csv

# Ensure required columns exist
required_cols = ["title", "overview", "genres", "keywords", "cast", "crew"]
for col in required_cols:
    if col not in movies.columns:
        movies[col] = ""  # Fill missing columns with empty strings

# Fill missing overviews
movies["overview"] = movies["overview"].fillna("")

# -----------------------------
# Helper functions
# -----------------------------
def convert(obj):
    """Convert stringified list of dicts to list of names"""
    try:
        return [i["name"] for i in ast.literal_eval(obj)]
    except:
        return []

def collapse(L):
    """Remove spaces in list items"""
    return [i.replace(" ", "") for i in L]

# -----------------------------
# Process columns
# -----------------------------
movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)
movies["cast"] = movies["cast"].apply(lambda x: convert(x)[:3])  # take top 3 cast
movies["crew"] = movies["crew"].apply(lambda x: [i for i in convert(x) if "Director" in i])

# Split overview into words
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces in all list columns
movies["genres"] = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"] = movies["cast"].apply(collapse)
movies["crew"] = movies["crew"].apply(collapse)

# -----------------------------
# Create "tags" for recommendation
# -----------------------------
movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

# Stemming
ps = PorterStemmer()
movies["tags"] = movies["tags"].apply(lambda x: " ".join([ps.stem(i) for i in x.split()]))

# -----------------------------
# Vectorization and similarity
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# -----------------------------
# Recommendation function
# -----------------------------
def recommend(movie):
    if movie not in movies["title"].values:
        print(f"Movie '{movie}' not found!")
        return
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    recommended = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    print("Recommended Movies:")
    for i in recommended:
        print(movies.iloc[i[0]].title)

# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    recommend("Avatar")
