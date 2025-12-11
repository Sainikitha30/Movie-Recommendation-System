# 🎬 Movie Recommendation System using Machine Learning

This project is a **Content-Based Movie Recommendation System** built using Python.  
It uses TMDB movie metadata and recommends similar movies using **Count Vectorizer + Cosine Similarity**.

---

# 📌 Full Source Code

Below is the complete Python code extracted from the Jupyter Notebook:

```python
# ---- Movie Recommendation System ----
import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets on title
movies = movies.merge(credits, on="title")

# Select useful columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# Drop missing values
movies.dropna(inplace=True)

# Convert stringified lists to python objects
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i["name"])
            count += 1
        else:
            break
    return L

movies["cast"] = movies["cast"].apply(convert3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L

movies["crew"] = movies["crew"].apply(fetch_director)

# Clean overview
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# Remove spaces
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies["genres"] = movies["genres"].apply(collapse)
movies["keywords"] = movies["keywords"].apply(collapse)
movies["cast"] = movies["cast"].apply(collapse)
movies["crew"] = movies["crew"].apply(collapse)

# Create "tags"
movies["tags"] = (
    movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
)

new_df = movies[["movie_id", "title", "tags"]]
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df["tags"] = new_df["tags"].apply(stem)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("Recommended Movies:")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Example run
recommend("Avatar")
