def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {c['full_id']}\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )

def extract_articles(chunks: list[dict]) -> tuple[list[str], list[str]]:
    seen_docs, seen_articles = set(), set()
    docs, articles = [], []
    for chunk in chunks:
        doc_id = f"{chunk['law_id']}|{chunk['law_name']}"
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            docs.append(doc_id)
        article_id = chunk["full_id"]
        if article_id not in seen_articles:
            seen_articles.add(article_id)
            articles.append(article_id)
    return docs, articles

def format_citation(chunk: dict) -> str:
    return f"{chunk['article_id']}, {chunk['law_name']} ({chunk['law_id']})"

def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    seen, result = set(), []
    for c in chunks:
        if c["full_id"] not in seen:
            seen.add(c["full_id"])
            result.append(c)
    return result