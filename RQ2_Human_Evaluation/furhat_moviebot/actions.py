# actions.py
import movie_kb_tmdb as movie_kb
import re

# ------------------------------------------------------------------
# Retrieval action (returns documents with metadata)
# ------------------------------------------------------------------
async def retrieve_movies(**kwargs):
    query = kwargs.get("query", "")
    print(f"🔍 Retrieving for: '{query}'")
    if not query:
        return []
    
    results = movie_kb.query_movies(query, n_results=5)
    if not results or 'documents' not in results or not results['documents'][0]:
        print("❌ No documents retrieved from ChromaDB")
        return []
    
    docs = results['documents'][0]
    # ChromaDB returns metadata as a list of dicts for each result
    metadatas = results.get('metadatas', [None])[0] if results.get('metadatas') else [{}] * len(docs)
    if metadatas is None:
        metadatas = [{}] * len(docs)
    
    print(f"📚 Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"   {i+1}: {doc[:200]}...")
    
    # Combine text and metadata
    retrieved = []
    for doc, meta in zip(docs, metadatas):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "score": 0.8  # placeholder, can be ignored
        })
    return retrieved

# ------------------------------------------------------------------
# Filtering action: keeps only movies whose director matches the query
# ------------------------------------------------------------------
def extract_director(query):
    """Simple extraction: look for director mentions."""
    patterns = [
        r"(?:director|directed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"movies?\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:films?|movies?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

async def filter_by_director(**kwargs):
    documents = kwargs.get("documents", [])
    query = kwargs.get("query", "")
    
    target_director = extract_director(query)
    if not target_director:
        # No director mentioned, keep all
        return documents
    
    filtered = []
    for doc in documents:
        meta = doc.get("metadata", {})
        director = meta.get("director", "")
        if director and target_director.lower() in director.lower():
            filtered.append(doc)
    
    print(f"🎬 Filtered to {len(filtered)} documents for director '{target_director}'")
    # If after filtering we have nothing, return the original (maybe the query was ambiguous)
    return filtered if filtered else documents