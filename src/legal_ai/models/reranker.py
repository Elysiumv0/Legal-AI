import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 2048
class Reranker:
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
    @torch.no_grad()
    def score(self, query: str, chunks: list[dict]) -> list[float]:
        pairs = [[query, c["text"]] for c in chunks]
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        logits = self.model(**encoded).logits
        if logits.dim() > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        scores = torch.sigmoid(logits.float()).cpu().flatten().tolist()
        return scores

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> list[dict]:
        all_scores = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            scores = self.score(query, batch)
            all_scores.extend(scores)
        for chunk, score in zip(chunks, all_scores):
            chunk["rerank_score"] = score
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

if __name__ == "__main__":
    mock_chunks = [
        {
            "full_id": "59/2020/QH14|Luật DN|Điều 47",
            "text": "Điều 47. Góp vốn thành lập công ty và cấp giấy chứng nhận phần vốn góp\n1. Vốn điều lệ của công ty trách nhiệm hữu hạn hai thành viên trở lên khi đăng ký thành lập doanh nghiệp là tổng giá trị phần vốn góp của các thành viên cam kết góp vào công ty.",
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp",
            "article_id": "Điều 47",
        },
        {
            "full_id": "59/2020/QH14|Luật DN|Điều 93",
            "text": "Điều 93. Tiêu chuẩn và điều kiện của thành viên Hội đồng thành viên\n1. Không thuộc đối tượng quy định tại khoản 2 Điều 17 của Luật này.",
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp",
            "article_id": "Điều 93",
        },
        {
            "full_id": "59/2020/QH14|Luật DN|Điều 79",
            "text": "Điều 79. Điều kiện và thủ tục thành lập công ty trách nhiệm hữu hạn một thành viên\n1. Công ty trách nhiệm hữu hạn một thành viên là doanh nghiệp do một tổ chức hoặc một cá nhân làm chủ sở hữu.",
            "law_id": "59/2020/QH14",
            "law_name": "Luật Doanh nghiệp",
            "article_id": "Điều 79",
        },
    ]

    reranker = Reranker()
    query = "điều kiện thành lập công ty TNHH một thành viên"
    print(f"Query: {query}\n")
    print("Before rerank:")
    for i, c in enumerate(mock_chunks):
        print(f"  {i+1}. {c['article_id']}")
    reranked = reranker.rerank(query, mock_chunks, top_k=3)
    print("\nAfter rerank:")
    for r in reranked:
        print(f"  [{r['rerank_score']:.4f}] {r['article_id']}")
        print(f"  {r['text'][:80]}")