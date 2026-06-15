import re
import json
from typing import List, Dict, Tuple, Any
from legal_ai.models.llm import LLMClient
from legal_ai.rag.schemas import RAGResult
def parse_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

class LegalCitationExtractor:
    def __init__(
        self,
        llm: LLMClient,
        kg_graph: Dict[str, List[str]],
        chunks: List[Dict[str, Any]]
    ):
        self.llm = llm
        self.kg = kg_graph
        self.chunks = chunks
        self.chunk_map = {c["full_id"]: c for c in chunks}
        self.law_article_map = {}
        for c in chunks:
            key = (c["law_id"], c["article_id"])
            if key not in self.law_article_map:
                self.law_article_map[key] = c

    def extract(
        self,
        answer: str,
        contexts: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        llm_citations = self._extract_with_llm(answer)
        validated_citations = self._validate_with_corpus(llm_citations)
        supplemented_citations = self._supplement_with_kg(
            validated_citations, contexts
        )
        return self._format_docs(supplemented_citations), \
               self._format_articles(supplemented_citations)

    def _extract_with_llm(self, answer: str) -> List[Dict[str, Any]]:
        prompt = (
            "Trích xuất TẤT CẢ các điều luật được đề cập trong câu trả lời dưới dạng JSON:\n"
            "{\n"
            "  \"citations\": [\n"
            "    {\n"
            "      \"law_id\": \"59/2020/QH14\",\n"
            "      \"article_id\": \"Điều 47\",\n"
            "      \"law_name\": \"Luật Doanh nghiệp 2020\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Câu trả lời: {answer}"
        )
        response = self.llm.chat([
            {"role": "user", "content": prompt}
        ])
        try:
            parsed = parse_json(response)
            citations = parsed.get("citations", [])
            if citations:
                return citations
        except Exception:
            pass
        return self._extract_with_regex(answer)

    def _validate_with_corpus(
        self,
        citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        validated = []
        for c in citations:
            if self._is_valid_citation(c):
                validated.append(c)
        return validated

    def _is_valid_citation(self, citation: Dict[str, Any]) -> bool:
        law_id = citation.get("law_id", "")
        article_id = citation.get("article_id", "")
        if not law_id or not article_id:
            return False
        return (law_id, article_id) in self.law_article_map

    def _supplement_with_kg(
        self,
        citations: List[Dict[str, Any]],
        contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        top_contexts = contexts[:3]
        existing_keys = {
            (c["law_id"], c["article_id"]) for c in citations
        }
        for c in top_contexts:
            full_id = c.get("full_id", "")
            if full_id not in self.kg:
                continue
            for ref_id in list(self.kg[full_id])[:2]:
                ref_chunk = self.chunk_map.get(ref_id)
                if not ref_chunk:
                    continue
                ref_key = (ref_chunk["law_id"], ref_chunk["article_id"])
                if ref_key not in existing_keys:
                    citations.append({
                        "law_id": ref_chunk["law_id"],
                        "article_id": ref_chunk["article_id"],
                        "law_name": ref_chunk["law_name"]
                    })
                    existing_keys.add(ref_key)
                    if len(citations) >= 7:
                        return citations
        return citations

    def _format_docs(
        self,
        citations: List[Dict[str, Any]]
    ) -> List[str]:
        docs = []
        seen = set()
        for c in citations:
            key = c['law_id']
            if key not in seen:
                seen.add(key)
                docs.append(f"{c['law_id']}|{c['law_name']}")
        return docs

    def _format_articles(
        self,
        citations: List[Dict[str, Any]]
    ) -> List[str]:
        articles = []
        seen = set()
        for c in citations:
            key = (c['law_id'], c['article_id'])
            if key not in seen:
                seen.add(key)
                articles.append(f"{c['law_id']}|{c['law_name']}|{c['article_id']}")
        return articles

    def _extract_with_regex(self, answer: str) -> List[Dict[str, Any]]:
        citations = []
        seen = set()
        pattern = r'[Đđ]iều\s+(\d+[a-z]?)[^\n,]*?(?:[Ll]uật|[Nn]ghị\s+[Đđ]ịnh|[Bb]ộ\s+[Ll]uật|[Tt]hông\s+[Tt]ư)\s+([^\n,]+)'
        for m in re.finditer(pattern, answer):
            article_num = m.group(1)
            law_name = m.group(2).strip()
            key = (article_num, law_name)
            if key in seen:
                continue
            seen.add(key)
            law_id = self._find_law_id(law_name, article_num)
            if law_id:
                citations.append({
                    "law_id": law_id,
                    "article_id": f"Điều {article_num}",
                    "law_name": law_name
                })
        return citations

    def _find_law_id(self, law_name: str, article_num: str) -> str | None:
        law_name_lower = law_name.lower()
        for key, chunk in self.law_article_map.items():
            chunk_law_id, chunk_article_id = key
            chunk_law_name = chunk.get("law_name", "").lower()
            if article_num not in chunk_article_id:
                continue
            if law_name_lower in chunk_law_name or chunk_law_name in law_name_lower:
                return chunk_law_id
        return None