from abc import ABC, abstractmethod
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        ...

    @abstractmethod
    def get_citation(self, chunk: dict) -> str:
        ...