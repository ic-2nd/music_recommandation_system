# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the Dataset
# Replace the file path with the actual path to your downloaded CSV file
data = pd.read_csv(
    "tcc_ceds_music.csv"
)  # Metadata about songs and user interactions
data.head()

# Step 3: Exploratory Data Analysis (EDA)
# Distribution of Songs by Genre
plt.figure(figsize=(10, 6))
sns.countplot(y="genre", data=data, order=data["genre"].value_counts().index[:10])
plt.title("Top 10 Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()

# Top Artists by Song Count
top_artists = data.groupby("artist_name").size().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
plt.title("Top 10 Artists by Number of Songs")
plt.xlabel("Number of Songs")
plt.ylabel("Artist Name")
plt.show()

# Step 4: Preprocessing the Data
# Combine song metadata into a single feature for similarity computation
data["combined_features"] = (
    data["genre"].fillna("")
    + " "
    + data["artist_name"].fillna("")
    + " "
    + data["track_name"].fillna("")
)


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["combined_features"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 5: Build the Recommendation Function
def get_recommendations(song_title, data, cosine_sim, top_n=10):
    # Get the index of the song that matches the title
    idx = data[data["track_name"] == song_title].index
    if len(idx) == 0:
        print("Song not found in the dataset.")
        return

    idx = idx[0]

    # Get similarity scores for all songs
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort songs based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N most similar songs
    sim_scores = sim_scores[1 : top_n + 1]  # Exclude the song itself
    song_indices = [i[0] for i in sim_scores]

    # Return recommended songs
    recommendations = data.iloc[song_indices]
    return recommendations

recommended_songs = get_recommendations("cry", data, cosine_sim, top_n=10)


print(recommended_songs[["track_name", "artist_name", "genre"]])

plt.figure(figsize=(10, 6))
sns.barplot(y="track_name", x="artist_name", data=recommended_songs, palette="coolwarm")
plt.title('Recommended Songs Similar to "Cry"')
plt.xlabel("Artist Name")
plt.ylabel("Song Name")
plt.show()