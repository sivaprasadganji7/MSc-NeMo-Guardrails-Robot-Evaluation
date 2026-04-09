"""
Data Ingestion Pipeline for TMDB 5000 Movies Dataset.

Downloads the dataset from Kaggle (if not present), processes the CSV files,
and ingests movie documents into Qdrant with hybrid dense+sparse embeddings.

Usage:
    python -m rag_core.data_ingestion
"""

import json
import os
import logging

import pandas as pd
from dotenv import load_dotenv, find_dotenv

from rag_core.hybrid_qdrant_operations import HybridQdrantOperations
from utils.decorators import compute_execution_time

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class TMDBDataIngestion:
    """Processes TMDB 5000 CSVs and ingests into Qdrant."""

    def __init__(self):
        self.movies_path = os.environ.get("TMDB_MOVIES_CSV", "data/tmdb_5000_movies.csv")
        self.credits_path = os.environ.get("TMDB_CREDITS_CSV", "data/tmdb_5000_credits.csv")
        self.qdrant_ops = HybridQdrantOperations()

    @compute_execution_time
    def load_and_merge(self) -> pd.DataFrame:
        """Load both TMDB CSVs and merge on movie id."""
        print("📂 Loading TMDB movies CSV...")
        movies_df = pd.read_csv(self.movies_path)

        print("📂 Loading TMDB credits CSV...")
        credits_df = pd.read_csv(self.credits_path)

        # Merge on the movie_id / id column
        # The credits CSV has 'movie_id', the movies CSV has 'id'
        merged = movies_df.merge(
            credits_df, left_on="id", right_on="movie_id", how="left", suffixes=("", "_credits")
        )

        print(f"✅ Merged dataset: {len(merged)} movies")
        return merged

    @compute_execution_time
    def process_dataframe(self, df: pd.DataFrame) -> tuple[list[str], list[dict]]:
        """
        Transform the merged DataFrame into documents and metadata
        suitable for Qdrant ingestion.
        """
        documents = []
        metadata = []

        for _, row in df.iterrows():
            # ── Build the document text (used for embedding) ──
            title = str(row.get("title", "Unknown"))
            overview = str(row.get("overview", ""))

            # Parse JSON columns safely
            genres = self._parse_json_names(row.get("genres", "[]"))
            keywords = self._parse_json_names(row.get("keywords", "[]"))
            cast = self._parse_cast(row.get("cast", "[]"), top_n=5)
            director = self._extract_director(row.get("crew", "[]"))

            # Compose a rich text document for embedding
            doc_parts = [
                f"Title: {title}",
                f"Overview: {overview}" if overview and overview != "nan" else "",
                f"Genres: {', '.join(genres)}" if genres else "",
                f"Keywords: {', '.join(keywords)}" if keywords else "",
                f"Cast: {', '.join(cast)}" if cast else "",
                f"Director: {director}" if director else "",
            ]
            document = "\n".join(part for part in doc_parts if part)

            if not document.strip():
                continue

            documents.append(document)

            # ── Build metadata (stored as payload in Qdrant) ──
            meta = {
                "title": title,
                "overview": overview if overview != "nan" else "",
                "genres": genres,
                "keywords": keywords,
                "cast": ", ".join(cast),
                "director": director,
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(row.get("vote_average", 0)),
                "vote_count": int(row.get("vote_count", 0)),
                "budget": float(row.get("budget", 0)),
                "revenue": float(row.get("revenue", 0)),
                "runtime": float(row.get("runtime", 0)) if pd.notna(row.get("runtime")) else 0,
                "original_language": str(row.get("original_language", "")),
                "popularity": float(row.get("popularity", 0)),
            }
            metadata.append(meta)

        print(f"✅ Processed {len(documents)} movie documents")
        return documents, metadata

    @compute_execution_time
    def ingest(self):
        """Full pipeline: load → process → create collection → insert."""
        df = self.load_and_merge()
        documents, metadata_list = self.process_dataframe(df)

        print("\n🔧 Setting up Qdrant collection...")
        self.qdrant_ops.create_collection()

        print(f"\n📤 Ingesting {len(documents)} documents into Qdrant...")
        self.qdrant_ops.insert_documents_from_lists(
            documents=documents, metadata=metadata_list
        )

        print("\n✅ Ingestion complete!")
        print(f"   Collection: {self.qdrant_ops.collection_name}")
        print(f"   Documents:  {len(documents)}")

    # ── JSON Parsing Helpers ────────────────────────────────────
    @staticmethod
    def _parse_json_names(raw: str, key: str = "name") -> list[str]:
        """Parse a JSON array of objects and extract the 'name' field."""
        try:
            items = json.loads(str(raw)) if isinstance(raw, str) else raw
            if isinstance(items, list):
                return [item[key] for item in items if isinstance(item, dict) and key in item]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _parse_cast(raw: str, top_n: int = 5) -> list[str]:
        """Extract top N actor names from the cast JSON."""
        try:
            cast_list = json.loads(str(raw)) if isinstance(raw, str) else raw
            if isinstance(cast_list, list):
                return [
                    member["name"]
                    for member in cast_list[:top_n]
                    if isinstance(member, dict) and "name" in member
                ]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _extract_director(raw: str) -> str:
        """Extract the director name from the crew JSON."""
        try:
            crew_list = json.loads(str(raw)) if isinstance(raw, str) else raw
            if isinstance(crew_list, list):
                for member in crew_list:
                    if isinstance(member, dict) and member.get("job") == "Director":
                        return member.get("name", "")
        except (json.JSONDecodeError, TypeError):
            pass
        return ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  TMDB 5000 → Qdrant Ingestion Pipeline")
    print("=" * 60)
    ingestion = TMDBDataIngestion()
    ingestion.ingest()