from legal_ai.rag.pipeline import StandardRAGPipeline
from legal_ai.rag.schemas import RAGResult
from legal_ai.retrieval.legal_rag import LegalRetriever
from legal_ai.rag.citation_extractor import LegalCitationExtractor


class SelfRAGPipeline:
    def __init__(
        self,
        rag_pipeline: StandardRAGPipeline,
        legal_retriever: LegalRetriever | None = None,
        citation_extractor: LegalCitationExtractor | None = None,
    ):
        self.rag = rag_pipeline
        self.legal_retriever = legal_retriever
        self.citation_extractor = citation_extractor

    def run(self, question: str) -> RAGResult:
        if self.legal_retriever:
            contexts = self.legal_retriever.retrieve(question)
        else:
            contexts = self.rag.retrieve(question)
        prompt = self.rag.augment(question, contexts)
        answer = self.rag.generate(prompt)
        if self.citation_extractor:
            docs, articles = self.citation_extractor.extract(answer, contexts)
        else:
            from legal_ai.rag.citations import extract_articles
            docs, articles = extract_articles(contexts)
        top_scores = [
            c.get("score", c.get("rrf_score", 0.0))
            for c in contexts[:3]
        ]
        
        return RAGResult(
            question=question,
            answer=answer,
            relevant_docs=docs,
            relevant_articles=articles,
            chunks_used=contexts,
            top_scores=top_scores,
            routed_to_agent=False,
        )