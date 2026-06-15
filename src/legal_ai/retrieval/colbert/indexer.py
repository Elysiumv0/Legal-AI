"""
ColBERT Indexer — build và quản lý ColBERT index cho legal chunks.
Dùng RAGatouille wrapper + jina-colbert-v2 (multilingual, hỗ trợ tiếng Việt).
"""

import json
import shutil
from pathlib import Path

from ragatouille import RAGPretrainedModel

MODEL_NAME = "jinaai/jina-colbert-v2"
INDEX_NAME = "legal_colbert"
INDEX_ROOT = "data/colbert_index"
MAX_DOC_LENGTH = 512


def load_chunks(chunks_path: str) -> list[dict]:
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def prepare_documents(chunks: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Chuẩn bị 3 list song song cho RAGatouille:
    - documents:  nội dung text (prepend header để ColBERT match token-level chính xác hơn)
    - doc_ids:    chunk_id làm key
    - metadatas:  metadata để trả về lúc search
    """
    documents = []
    doc_ids = []
    metadatas = []

    for c in chunks:
        header = c.get("header", "")
        text = c["text"]

        # Prepend header nếu text chưa chứa header
        # → giúp ColBERT token matching bắt được "Điều 47" ngay đầu
        if header and not text.startswith(header):
            text = f"{header}\n{text}"

        documents.append(text)
        doc_ids.append(c["chunk_id"])
        metadatas.append({
            "full_id":    c["full_id"],
            "law_id":     c["law_id"],
            "law_name":   c["law_name"],
            "article_id": c["article_id"],
            "header":     header,
            "khoan":      c.get("khoan"),
        })

    return documents, doc_ids, metadatas


def build_index(
    chunks_path: str = "data/processed/chunks.json",
    model_name: str = MODEL_NAME,
    index_name: str = INDEX_NAME,
    index_root: str = INDEX_ROOT,
    max_document_length: int = MAX_DOC_LENGTH,
    recreate: bool = False,
) -> Path:
    """
    Build ColBERT index.

    Returns:
        Path tới index đã build.
    """
    index_path = Path(index_root) / index_name

    if index_path.exists() and not recreate:
        print(f"Index đã tồn tại: {index_path}")
        print("Dùng recreate=True để tạo lại.")
        return index_path

    if index_path.exists() and recreate:
        print(f"Xoá index cũ: {index_path}")
        shutil.rmtree(index_path)

    chunks = load_chunks(chunks_path)
    documents, doc_ids, metadatas = prepare_documents(chunks)

    print(f"Building ColBERT index: model={model_name}, chunks={len(documents)}")
    model = RAGPretrainedModel.from_pretrained(model_name)

    model.index(
        collection=documents,
        document_ids=doc_ids,
        document_metadatas=metadatas,
        index_name=index_name,
        index_root=index_root,
        max_document_length=max_document_length,
        split_documents=False,  # chunks đã được split sẵn, không cắt thêm
    )

    print(f"ColBERT index built: {index_path}")
    return index_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build ColBERT index cho legal chunks")
    parser.add_argument(
        "--chunks", type=str,
        default="data/processed/chunks.json",
        help="Đường dẫn tới file chunks.json",
    )
    parser.add_argument("--recreate", action="store_true", help="Xoá và tạo lại index")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="ColBERT model name")
    args = parser.parse_args()

    build_index(
        chunks_path=args.chunks,
        model_name=args.model,
        recreate=args.recreate,
    )
