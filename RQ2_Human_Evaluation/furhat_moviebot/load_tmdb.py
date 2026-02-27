import os
import ast
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import kagglehub

# Configuration
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies_tmdb"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL
)

def parse_genres(genres_str):
    """Convert genres JSON string to comma-separated names."""
    try:
        genres = ast.literal_eval(genres_str)
        return ', '.join([g['name'] for g in genres])
    except:
        return ''

def parse_cast(cast_str):
    """Extract top 5 actor names."""
    try:
        cast = ast.literal_eval(cast_str)
        return ', '.join([c['name'] for c in cast[:5]])
    except:
        return ''

def parse_crew(crew_str):
    """Extract director name."""
    try:
        crew = ast.literal_eval(crew_str)
        directors = [c['name'] for c in crew if c['job'] == 'Director']
        return directors[0] if directors else ''
    except:
        return ''

def load_tmdb_to_db(limit=None):
    """Download TMDB dataset locally, then load into ChromaDB."""
    print("Downloading TMDB movie metadata...")
    # Download the entire dataset to a local cache folder
    dataset_path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Build paths to the CSV files
    movies_csv = os.path.join(dataset_path, "tmdb_5000_movies.csv")
    credits_csv = os.path.join(dataset_path, "tmdb_5000_credits.csv")
    
    # Read CSVs with proper encoding
    df = pd.read_csv(movies_csv, encoding='latin-1')
    credits_df = pd.read_csv(credits_csv, encoding='latin-1')
    
    print("Credits DataFrame columns:", credits_df.columns.tolist())
    print("Movies DataFrame columns:", df.columns.tolist())
    
    # Determine merge keys
    left_key = 'id'  # movies has 'id'
    if 'movie_id' in credits_df.columns:
        right_key = 'movie_id'
    elif 'id' in credits_df.columns:
        right_key = 'id'
    else:
        raise KeyError(f"Credits has neither 'movie_id' nor 'id'. Columns: {credits_df.columns.tolist()}")
    
    # Merge the two dataframes
    df = df.merge(credits_df, left_on=left_key, right_on=right_key, how='left')
    print(f"Merged on {left_key} (movies) and {right_key} (credits)")
    
    # Optionally limit number of movies
    if limit:
        df = df.head(limit)
    
    # Prepare documents and metadata
    documents = []
    ids = []
    metadatas = []
    
    for idx, row in df.iterrows():
        # Extract year from release_date
        year = ''
        if pd.notna(row['release_date']):
            try:
                year = str(pd.to_datetime(row['release_date']).year)
            except:
                year = ''
        
        # Parse genres and keywords
        genres = parse_genres(row['genres']) if pd.notna(row['genres']) else ''
        keywords = parse_genres(row['keywords']) if pd.notna(row['keywords']) else ''
        
        # Get director (handle possible column name suffix after merge)
        director = ''
        crew_col = None
        for col in ['crew', 'crew_y', 'crew_x']:
            if col in row.index:
                crew_col = col
                break
        if crew_col and pd.notna(row[crew_col]):
            try:
                crew = ast.literal_eval(row[crew_col])
                directors = [c['name'] for c in crew if c['job'] == 'Director']
                director = directors[0] if directors else ''
            except:
                pass
        
        # Get cast (top 5)
        cast = ''
        cast_col = None
        for col in ['cast', 'cast_y', 'cast_x']:
            if col in row.index:
                cast_col = col
                break
        if cast_col and pd.notna(row[cast_col]):
            try:
                cast_list = ast.literal_eval(row[cast_col])
                cast = ', '.join([c['name'] for c in cast_list[:5]])
            except:
                pass
        
        # Get title (handle possible suffix)
        title_col = None
        for col in ['title', 'title_x', 'title_y']:
            if col in row.index:
                title_col = col
                break
        title = row[title_col] if title_col else "Unknown"
        
        # Create document text
        doc_text = f"Title: {title}\n"
        if director:
            doc_text += f"Director: {director}\n"
        if year:
            doc_text += f"Year: {year}\n"
        if genres:
            doc_text += f"Genres: {genres}\n"
        if keywords:
            doc_text += f"Keywords: {keywords}\n"
        if cast:
            doc_text += f"Cast: {cast}\n"
        if pd.notna(row['overview']) and row['overview']:
            doc_text += f"Overview: {row['overview']}\n"
        
        documents.append(doc_text)
        ids.append(f"tmdb_{row['id']}")
        metadatas.append({
            "title": title,
            "year": year,
            "director": director,
            "genres": genres
        })
    
    # Create or get ChromaDB collection
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef
        )
        print("Created new collection.")
    except:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Using existing collection.")
    
    # Add documents in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    print(f"Total movies added: {len(documents)}")

if __name__ == "__main__":
    # Set limit to e.g., 1000 to control cost/time; remove for full dataset
    load_tmdb_to_db(limit=None)