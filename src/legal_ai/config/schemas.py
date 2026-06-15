from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "legal-ai"
    environment: str = "local"


class PathConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    corpus_path: Path = Path("data/processed/corpus.json")
    chunks_path: Path = Path("data/processed/chunks.json")
    metadata_path: Path = Path("data/processed/metadata.json")


class IndexConfig(BaseModel):
    bm25_path: Path = Path("data/indexes/bm25/bm25.pkl")
    qdrant_path: Path = Path("data/indexes/qdrant")
    colbert_root: Path = Path("data/indexes/colbert")


class ModelRuntimeConfig(BaseModel):
    name: str
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=512, ge=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class LLMConfig(BaseModel):
    provider: Literal["local", "openai_compatible", "ollama"] = "local"
    model_name: str
    base_url: str | None = None
    max_new_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ModelConfig(BaseModel):
    embedding: ModelRuntimeConfig
    reranker: ModelRuntimeConfig
    llm: LLMConfig


class RetrievalConfig(BaseModel):
    top_k_bm25: int = Field(default=30, ge=1)
    top_k_dense: int = Field(default=30, ge=1)
    top_k_colbert: int = Field(default=20, ge=1)
    top_k_rerank: int = Field(default=10, ge=1)
    use_bm25: bool = True
    use_dense: bool = True
    use_colbert: bool = False


class CacheConfig(BaseModel):
    enabled: bool = True
    long_term_path: Path = Path("data/cache/long_term.json")
    ttl_days: int = Field(default=30, ge=1)
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)


class Settings(BaseModel):
    project: ProjectConfig = ProjectConfig()
    paths: PathConfig = PathConfig()
    indexes: IndexConfig = IndexConfig()
    models: ModelConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    cache: CacheConfig = CacheConfig()


class DataSource(BaseModel):
    path: Path
    law_id: str
    law_name: str
    document_type: Literal["law", "decree", "circular", "other"] = "other"
    issued_date: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    status: Literal["active", "amended", "repealed", "unknown"] = "unknown"


class DataSources(BaseModel):
    sources: list[DataSource] = []


class AgentNodeConfig(BaseModel):
    enabled: bool = True


class ResearcherConfig(AgentNodeConfig):
    max_sub_queries: int = Field(default=3, ge=1)
    top_k_retrieve: int = Field(default=20, ge=1)
    top_k_rerank: int = Field(default=10, ge=1)


class CriticConfig(AgentNodeConfig):
    require_json: bool = True
    default_sufficient_on_parse_error: bool = False


class DrafterConfig(AgentNodeConfig):
    require_citations: bool = True


class ReviewerConfig(AgentNodeConfig):
    citation_check: bool = True


class AgentGraphConfig(BaseModel):
    entry_point: str = "researcher"
    max_retrieval_attempts: int = Field(default=3, ge=1)


class AgentGraphSettings(BaseModel):
    graph: AgentGraphConfig = AgentGraphConfig()
    researcher: ResearcherConfig = ResearcherConfig()
    critic: CriticConfig = CriticConfig()
    drafter: DrafterConfig = DrafterConfig()
    reviewer: ReviewerConfig = ReviewerConfig()
