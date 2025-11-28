# Music Recommendation System 🎵

A **content-based music recommendation system** built in Python that suggests songs similar to a given track using song metadata such as genre, artist, and track name. The system leverages **TF-IDF vectorization** and **cosine similarity** to provide personalized recommendations.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)

---

## Features
- Recommends songs similar to a given track.
- Uses **content-based filtering** on song metadata.
- Visualizes:
  - Top genres
  - Top artists
  - Recommended songs
- Easy to extend for larger datasets.

---

## Dataset
The system requires a CSV file with metadata about songs, including at least the following columns:

- `track_name`: Name of the song
- `artist_name`: Artist performing the song
- `genre`: Genre of the song

Example file used: `tcc_ceds_music.csv`

## Installation
1. Clone the repository:
`git clone https://github.com/ic_2nd/music-recommendation-system.git`
`cd music-recommendation-system`

Install dependencies:
`pip install pandas numpy scikit-learn matplotlib seaborn`
Place your CSV dataset in the project directory.

Update the file path in the script:
`data = pd.read_csv("/path/to/your/tcc_ceds_music.csv")`

Run the script:
`python music_recommender.py`
