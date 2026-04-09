# movie_kb_tmdb.py
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()
import os


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movies_tmdb"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL
)

def get_collection():
    """Retrieve the movie collection."""
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
    except Exception as e:
        print(f"Error getting collection: {e}")
        return None

def query_movies(query_text, n_results=3):
    """Query the movie knowledge base and return top results."""
    collection = get_collection()
    if not collection:
        return None
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results