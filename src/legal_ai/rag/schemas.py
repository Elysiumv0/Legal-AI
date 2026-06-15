from dataclasses import dataclass, field
@dataclass
class RAGResult:
    question:          str
    answer:            str
    relevant_docs:     list[str]
    relevant_articles: list[str]
    chunks_used:       list[dict] = field(default_factory=list)
    top_scores:        list[float] = field(default_factory=list)
    routed_to_agent:   bool = False

@dataclass
class SubmitResult:
    id:                int
    question:          str
    answer:            str
    relevant_docs:     list[str]
    relevant_articles: list[str]
    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "question":          self.question,
            "answer":            self.answer,
            "relevant_docs":     self.relevant_docs,
            "relevant_articles": self.relevant_articles,
        }