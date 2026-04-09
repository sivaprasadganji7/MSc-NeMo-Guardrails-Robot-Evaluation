"""
Hybrid Qdrant Operations for TMDB 5000 Movies.

Uses dense (snowflake-arctic-embed-s) + sparse (Splade) embeddings
for hybrid retrieval over movie documents.
"""

import os
import logging
from typing import Any

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv, find_dotenv

from utils.decorators import compute_execution_time

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class HybridQdrantOperations:
    """Manages Qdrant collection with hybrid dense+sparse vector search."""

    DENSE_MODEL = "snowflake/snowflake-arctic-embed-s"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
    COLLECTION_NAME = "tmdb-5000-hybrid-collection"

    def __init__(self):
        qdrant_url = os.environ.get("QDRANT_API_BASE", "http://localhost:6333")
        qdrant_key = os.environ.get("QDRANT_API_KEY", "")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key if qdrant_key else None,
        )
        self.collection_name = self.COLLECTION_NAME

        # Configure embedding models
        self.client.set_model(self.DENSE_MODEL)
        self.client.set_sparse_model(self.SPARSE_MODEL)

    @compute_execution_time
    def create_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        if self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists, skipping creation.")
            print(f"  ℹ️  Collection '{self.collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.client.get_fastembed_vector_params(),
            sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(on_disk=True),
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=5,
                indexing_threshold=0,  # Disable indexing during bulk insert
            ),
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True),
            ),
            shard_number=2,
        )
        logger.info(f"Created collection '{self.collection_name}'")
        print(f"  ✅ Created collection '{self.collection_name}'")

    @compute_execution_time
    def insert_documents_from_lists(self, documents: list[str], metadata: list[dict]):
        """
        Insert documents with metadata into Qdrant.
        Uses fastembed to automatically generate dense + sparse vectors.
        Processes in small batches to avoid out-of-memory errors.
        """
        BATCH_SIZE = 100  # Small batches to fit in RAM
        total = len(documents)

        for i in range(0, total, BATCH_SIZE):
            batch_docs = documents[i : i + BATCH_SIZE]
            batch_meta = metadata[i : i + BATCH_SIZE]
            self.client.add(
                collection_name=self.collection_name,
                documents=batch_docs,
                metadata=batch_meta,
                parallel=1,  # Single worker to save memory
            )
            print(f"  📦 Ingested batch {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} ({min(i + BATCH_SIZE, total)}/{total} docs)")

        self._optimize_after_insert()
        logger.info(f"Inserted {len(documents)} documents into '{self.collection_name}'")

    @compute_execution_time
    def hybrid_search(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform hybrid (dense + sparse) search.
        Returns metadata of the top-k most relevant documents.
        """
        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=text,
            limit=top_k,
        )

        results = [hit.metadata for hit in search_results]
        logger.info(f"Hybrid search for '{text[:50]}...' returned {len(results)} results")
        return results

    @compute_execution_time
    def filtered_search(
        self,
        text: str,
        genre: str | None = None,
        min_rating: float | None = None,
        min_year: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search with optional metadata filters.
        Useful for queries like "best action movies rated above 8".
        """
        conditions = []

        if genre:
            conditions.append(
                models.FieldCondition(
                    key="genres",
                    match=models.MatchAny(any=[genre]),
                )
            )

        if min_rating is not None:
            conditions.append(
                models.FieldCondition(
                    key="vote_average",
                    range=models.Range(gte=min_rating),
                )
            )

        if min_year:
            conditions.append(
                models.FieldCondition(
                    key="release_date",
                    range=models.Range(gte=min_year),
                )
            )

        query_filter = models.Filter(must=conditions) if conditions else None

        search_results = self.client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=query_filter,
            limit=top_k,
        )

        return [hit.metadata for hit in search_results]

    def _optimize_after_insert(self):
        """Re-enable indexing after bulk insert."""
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=30000),
        )

    def get_collection_info(self) -> dict:
        """Return collection stats for health checks."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }