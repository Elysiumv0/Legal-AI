import json
import re
from typing import Dict, List, Any

from legal_ai.agents.state import AgentState
from legal_ai.retrieval.legal_rag import LegalRetriever
from legal_ai.rag.citation_extractor import LegalCitationExtractor, parse_json
from legal_ai.models.llm import LLMClient


class LegalSelfRAG:
    def __init__(
        self,
        retriever: LegalRetriever,
        extractor: LegalCitationExtractor,
        llm: LLMClient,
    ):
        self.retriever = retriever
        self.extractor = extractor
        self.llm = llm
        self.max_iterations = 2  # Giới hạn số lần self-correct

    def run(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        contexts = state.get("retrieved_chunks", [])
        
        # Nếu chưa có contexts thì retrieve
        if not contexts:
            contexts = self.retriever.retrieve(question)
        
        for iteration in range(self.max_iterations):
            # BƯỚC 1: Extract citations
            docs, articles = self.extractor.extract("", contexts)
            
            # BƯỚC 2: Self-evaluation
            evaluation = self._evaluate(question, contexts, docs, articles)
            
            # BƯỚC 3: Tự sửa nếu cần
            if evaluation.get("is_sufficient", True):
                break  # Đủ rồi, thoát
            
            # Tạo query mới để tìm thêm
            new_query = self._generate_new_query(question, evaluation)
            if not new_query or new_query == question:
                break  # Không sinh được query mới
            
            # Retrieve thêm
            new_contexts = self.retriever.retrieve(new_query)
            
            # Merge contexts (deduplicate)
            existing_ids = {c.get("chunk_id") for c in contexts}
            for c in new_contexts:
                if c.get("chunk_id") not in existing_ids:
                    contexts.append(c)
        
        # BƯỚC 4: Final answer
        docs, articles = self.extractor.extract("", contexts)
        return {
            "relevant_docs": docs,
            "relevant_articles": articles,
            "retrieved_chunks": contexts,
        }

    def _evaluate(
        self,
        question: str,
        contexts: List[Dict[str, Any]],
        docs: List[str],
        articles: List[str]
    ) -> Dict[str, Any]:
        """Đánh giá xem context có đủ để trả lời câu hỏi không"""
        prompt = f"""Đánh giá xem các điều luật sau có đủ để trả lời câu hỏi không.

Câu hỏi: {question}

Các điều luật đã tìm được:
{chr(10).join(articles) if articles else "Không có"}

Trả về JSON:
{{
  "is_sufficient": true/false,
  "missing": "thông tin còn thiếu (nếu có)"
}}

JSON:"""
        
        try:
            response = self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            return parse_json(response)
        except Exception:
            return {"is_sufficient": True}  # Default: đủ

    def _generate_new_query(
        self,
        question: str,
        evaluation: Dict[str, Any]
    ) -> str:
        """Tự sinh query mới để tìm thêm thông tin"""
        missing = evaluation.get("missing", "")
        if not missing:
            return ""
        
        prompt = f"""Tạo 1 câu hỏi pháp lý ngắn gọn để tìm thông tin còn thiếu.

Câu hỏi gốc: {question}
Thông tin còn thiếu: {missing}

Trả về JSON:
{{
  "new_query": "câu hỏi mới"
}}

JSON:"""
        
        try:
            response = self.llm.chat([
                {"role": "user", "content": prompt}
            ])
            parsed = parse_json(response)
            return parsed.get("new_query", "")
        except Exception:
            return ""