"""
Metadata DB — Lưu và truy vấn metadata văn bản pháp luật
Dùng SQLite (đủ cho hackathon, không cần PostgreSQL)
"""

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass


DB_PATH = "data/cache/metadata.db"


@dataclass
class LawRecord:
    law_id:         str
    law_name:       str
    law_type:       str            # Luật, Nghị định, Thông tư
    issuer:         str | None     # Cơ quan ban hành
    issued_date:    str | None     # Ngày ban hành
    effective_date: str | None     # Ngày có hiệu lực
    status:         str            # Còn hiệu lực / Hết / Một phần
    parent_law_id:  str | None     # NĐ/TT hướng dẫn Luật nào
    total_articles: int = 0


class MetadataDB:
    """
    SQLite store cho metadata văn bản pháp luật.

    Dùng để:
    - Filter retrieval theo loại văn bản, hiệu lực
    - Tra cứu chéo: NĐ nào hướng dẫn Luật DN?
    - Agent check hiệu lực trước khi cite
    """

    def __init__(self, db_path: str = DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS laws (
                law_id          TEXT PRIMARY KEY,
                law_name        TEXT NOT NULL,
                law_type        TEXT,
                issuer          TEXT,
                issued_date     TEXT,
                effective_date  TEXT,
                status          TEXT DEFAULT 'Còn hiệu lực',
                parent_law_id   TEXT,
                total_articles  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS law_relations (
                child_law_id    TEXT,
                parent_law_id   TEXT,
                relation_type   TEXT,   -- "hướng dẫn", "sửa đổi", "thay thế"
                PRIMARY KEY (child_law_id, parent_law_id)
            );
        """)
        self.conn.commit()

    # ── Insert / Update ───────────────────────────────────────────────────────

    def upsert_law(self, record: LawRecord):
        self.conn.execute("""
            INSERT OR REPLACE INTO laws
            (law_id, law_name, law_type, issuer, issued_date,
             effective_date, status, parent_law_id, total_articles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.law_id, record.law_name, record.law_type,
            record.issuer, record.issued_date, record.effective_date,
            record.status, record.parent_law_id, record.total_articles,
        ))
        self.conn.commit()

    def add_relation(self, child_id: str, parent_id: str, relation: str):
        self.conn.execute("""
            INSERT OR IGNORE INTO law_relations
            (child_law_id, parent_law_id, relation_type)
            VALUES (?, ?, ?)
        """, (child_id, parent_id, relation))
        self.conn.commit()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_law(self, law_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM laws WHERE law_id = ?", (law_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_related_laws(self, law_id: str) -> list[dict]:
        """Lấy tất cả văn bản liên quan (hướng dẫn, sửa đổi)"""
        rows = self.conn.execute("""
            SELECT l.*, r.relation_type
            FROM laws l
            JOIN law_relations r ON l.law_id = r.child_law_id
            WHERE r.parent_law_id = ?
        """, (law_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_active_laws(self) -> list[dict]:
        """Chỉ lấy văn bản còn hiệu lực"""
        rows = self.conn.execute(
            "SELECT * FROM laws WHERE status = 'Còn hiệu lực'"
        ).fetchall()
        return [dict(r) for r in rows]

    def is_active(self, law_id: str) -> bool:
        """Check văn bản có còn hiệu lực không"""
        row = self.conn.execute(
            "SELECT status FROM laws WHERE law_id = ?", (law_id,)
        ).fetchone()
        return row["status"] == "Còn hiệu lực" if row else True

    # ── Bulk Load từ corpus ───────────────────────────────────────────────────

    def load_from_corpus(self, corpus_path: str = "data/processed/corpus.json"):
        """Load metadata từ corpus.json đã parse"""
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

        # Mapping hiệu lực thủ công — thực tế cần scrape từ vbpl.vn
        status_map = {
            "59/2020/QH14": "Còn hiệu lực",
            "04/2017/QH14": "Còn hiệu lực",
            "45/2019/QH14": "Còn hiệu lực",
            "91/2015/QH13": "Còn hiệu lực",
            "36/2005/QH11": "Còn hiệu lực",
            "14/2008/QH12": "Còn hiệu lực",
            "48/2024/QH15": "Còn hiệu lực",
            "01/2021/NĐ-CP": "Còn hiệu lực",
            "80/2021/NĐ-CP": "Còn hiệu lực",
            "145/2020/NĐ-CP": "Còn hiệu lực",
        }

        # Mapping quan hệ cha-con
        relations = [
            ("01/2021/NĐ-CP",  "59/2020/QH14",  "hướng dẫn"),
            ("80/2021/NĐ-CP",  "04/2017/QH14",  "hướng dẫn"),
            ("145/2020/NĐ-CP", "45/2019/QH14",  "hướng dẫn"),
        ]

        for doc in corpus:
            from legal_ai.ingestion.metadata import detect_law_type
            record = LawRecord(
                law_id=doc["law_id"],
                law_name=doc["law_name"],
                law_type=detect_law_type(doc["law_name"]),
                issuer=None,
                issued_date=None,
                effective_date=None,
                status=status_map.get(doc["law_id"], "Còn hiệu lực"),
                parent_law_id=None,
                total_articles=doc["total_articles"],
            )
            self.upsert_law(record)

        for child, parent, rel in relations:
            self.add_relation(child, parent, rel)

        print(f"✅ MetadataDB loaded {len(corpus)} laws")

    def close(self):
        self.conn.close()