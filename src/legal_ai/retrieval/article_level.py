from collections import defaultdict
from legal_ai.retrieval.dense.qdrant_store import VectorStore, ARTICLE_COLLECTION
from legal_ai.models.embeddings import EmbeddingModel

def build_article_docs(chunks: list[dict]) -> list[dict]:
    article_map = defaultdict(lambda: {"text": "", "meta": None})
    for chunk in chunks:
        fid = chunk["full_id"]
        article_map[fid]["text"] += "\n" + chunk["text"]
        if article_map[fid]["meta"] is None:
            article_map[fid]["meta"] = chunk
    docs = []
    for fid, data in article_map.items():
        doc = {**data["meta"],
               "chunk_id": fid,
               "text": data["text"].strip()}
        docs.append(doc)
    return docs

class ArticleLevelRetriever:
    def __init__(self, store: VectorStore, embedder: EmbeddingModel):
        self.store   = store
        self.embedder = embedder

    def build_index(self, chunks: list[dict], recreate=False):
        docs = build_article_docs(chunks)
        print(f"Building article-level index: {len(docs)} articles")
        dummy = self.embedder.encode(["test"])
        self.store.create_collection(
            dim=dummy.shape[1],
            collection_name=ARTICLE_COLLECTION,
            recreate=recreate
        )
        self.store.index(docs, self.embedder, collection_name=ARTICLE_COLLECTION)

    def retrieve(self, query: str, top_k: int = 50) -> list[dict]:
        vec = self.embedder.encode([query], is_query=True)[0]
        return self.store.search(vec, top_k=top_k, collection_name=ARTICLE_COLLECTION)