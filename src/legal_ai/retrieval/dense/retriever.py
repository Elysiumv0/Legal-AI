import json
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
from legal_ai.retrieval.dense.qdrant_store import VectorStore
from legal_ai.models.embeddings import EmbeddingModel
COLLECTION = "legal_chunks"
QDRANT_PATH = "data/indexes/qdrant"
BATCH_SIZE = 8
class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        query_vec = self.embedder.encode([query], is_query=True)[0]
        return self.store.search(query_vec, top_k=top_k)


def build_index(chunks_path: str = "data/processed/chunks.json", recreate: bool = False):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    embedder = EmbeddingModel()
    dummy_vec = embedder.encode(["test"])
    store = VectorStore()
    store.create_collection(dim=dummy_vec.shape[1], recreate=recreate)
    store.index(chunks, embedder)
    embedder.unload()
    return store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Xóa và tạo lại collection")
    parser.add_argument("--test", type=str, default=None, help="Test query sau khi index")
    args = parser.parse_args()
    store = build_index(recreate=args.recreate)
    if args.test:
        embedder = EmbeddingModel()
        retriever = DenseRetriever(store, embedder)
        results = retriever.retrieve(args.test, top_k=5)
        print(f"\nQuery: {args.test}")
        for result in results:
            print(f"[{result['score']:.3f}] {result['full_id']}")
            print(f"  {result['text'][:100]}")
            print()
