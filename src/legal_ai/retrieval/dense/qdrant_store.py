import numpy as np
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from legal_ai.models.embeddings import EmbeddingModel

CHUNK_COLLECTION   = "legal_chunks"
ARTICLE_COLLECTION = "legal_articles"
QDRANT_PATH        = "data/indexes/qdrant"
BATCH_SIZE         = 8


class VectorStore:
    def __init__(self, path: str = QDRANT_PATH):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path)

    def create_collection(
        self,
        dim: int,
        collection_name: str = CHUNK_COLLECTION,
        recreate: bool = False,
    ):
        exists = any(
            c.name == collection_name
            for c in self.client.get_collections().collections
        )
        if exists and not recreate:
            print(f"Collection '{collection_name}' đã tồn tại")
            return
        if exists:
            self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created, dim={dim}")

    def index(
        self,
        chunks: list[dict],
        embedder: EmbeddingModel,
        collection_name: str = CHUNK_COLLECTION,
        batch_size: int = BATCH_SIZE,
    ):
        total = len(chunks)
        print(f"Indexing {total} docs vào '{collection_name}'...")
        for i in tqdm(range(0, total, batch_size)):
            batch   = chunks[i: i + batch_size]
            texts   = [c["text"] for c in batch]
            vectors = embedder.encode(texts, is_query=False)
            points  = [
                PointStruct(
                    id=i + j,
                    vector=vectors[j].tolist(),
                    payload={
                        "chunk_id":        chunk["chunk_id"],
                        "text":            chunk["text"],
                        "law_id":          chunk["law_id"],
                        "law_name":        chunk["law_name"],
                        "article_id":      chunk["article_id"],
                        "header":          chunk.get("header", ""),
                        "full_id":         chunk["full_id"],
                        "khoan":           chunk.get("khoan"),
                        "chapter":         chunk.get("chapter"),
                        "chapter_name":    chunk.get("chapter_name"),
                        "section":         chunk.get("section"),
                        "law_type":        chunk.get("law_type"),
                        "status":          chunk.get("status"),
                        "context_prefix":  chunk.get("context_prefix"),
                    },
                )
                for j, chunk in enumerate(batch)
            ]
            self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {total} docs vào '{collection_name}'")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        collection_name: str = CHUNK_COLLECTION,
        law_id_filter: str | None = None,
    ) -> list[dict]:
        query_filter = None
        if law_id_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="law_id",
                    match=MatchValue(value=law_id_filter),
                )]
            )
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            {
                "chunk_id":       r.payload["chunk_id"],
                "text":           r.payload["text"],
                "law_id":         r.payload["law_id"],
                "law_name":       r.payload["law_name"],
                "article_id":     r.payload["article_id"],
                "full_id":        r.payload["full_id"],
                "chapter":        r.payload.get("chapter"),
                "chapter_name":   r.payload.get("chapter_name"),
                "law_type":       r.payload.get("law_type"),
                "status":         r.payload.get("status"),
                "context_prefix": r.payload.get("context_prefix"),
                "score":          r.score,
                "rank":           i + 1,
            }
            for i, r in enumerate(response.points)
        ]