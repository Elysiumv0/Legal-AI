from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from legal_ai.retrieval.hybrid.fusion import HybridRetriever
from legal_ai.models.reranker import Reranker
class LegalSearchInput(BaseModel):
    query:      str = Field(description="Câu hỏi hoặc từ khóa pháp lý cần tìm")
    top_k:      int = Field(default=20, description="Số lượng kết quả trả về")
    law_filter: str | None = Field(
        default=None,
        description="Lọc theo luật cụ thể, VD: '59/2020/QH14'"
    )

class ArticleLookupInput(BaseModel):
    law_id:     str = Field(description="Mã văn bản, VD: '59/2020/QH14'")
    article_num: str = Field(description="Số điều, VD: '47'")

def make_hybrid_search_tool(
    hybrid: HybridRetriever,
    reranker: Reranker,
) -> StructuredTool:
    def hybrid_search(
        query: str,
        top_k: int = 20,
        law_filter: str | None = None,
    ) -> list[dict]:
        results = hybrid.retrieve(query, top_k=top_k)
        if law_filter:
            results = [r for r in results if r["law_id"] == law_filter]
        return reranker.rerank(query, results, top_k=10)
    return StructuredTool(
        name="hybrid_search",
        description=(
            "Tìm kiếm điều luật pháp lý Việt Nam liên quan đến câu hỏi. "
            "Dùng khi cần tìm điều luật theo nội dung hoặc từ khóa."
        ),
        args_schema=LegalSearchInput,
        func=hybrid_search,
    )

def make_article_lookup_tool(chunks: list[dict]) -> StructuredTool:
    index = {}
    for chunk in chunks:
        key = (chunk["law_id"], chunk["article_id"].replace("Điều ", ""))
        if key not in index:
            index[key] = chunk

    def article_lookup(law_id: str, article_num: str) -> dict | None:
        return index.get((law_id, article_num))
    return StructuredTool(
        name="article_lookup",
        description=(
            "Tra cứu nội dung 1 Điều cụ thể khi đã biết mã văn bản và số điều. "
            "Dùng khi query đã có dạng 'Điều X Luật Y'."
        ),
        args_schema=ArticleLookupInput,
        func=article_lookup,
    )

class LegalToolbox:
    def __init__(
        self,
        hybrid: HybridRetriever,
        reranker: Reranker,
        chunks: list[dict],
    ):
        self.hybrid_search  = make_hybrid_search_tool(hybrid, reranker)
        self.article_lookup = make_article_lookup_tool(chunks)
        self.all_tools = [
            self.hybrid_search,
            self.article_lookup,
        ]
        
    def get_tool(self, name: str) -> StructuredTool | None:
        return next((t for t in self.all_tools if t.name == name), None)