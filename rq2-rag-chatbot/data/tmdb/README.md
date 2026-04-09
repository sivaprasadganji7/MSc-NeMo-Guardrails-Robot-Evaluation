# TMDB 5000 Movies Dataset

Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Place these files in this directory:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These CSV files are excluded from Git via .gitignore.
The ingestion script in `../src/retrieval/` will merge and index them into Qdrant.
