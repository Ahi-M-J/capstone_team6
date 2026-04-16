# src/api/v1/tools/fts_search_tool.py
from typing import List
from langchain_core.documents import Document
from src.core.db import get_db_conn


def fts_search(query: str, k: int = 25) -> List[Document]:
    sql = """
    SELECT
        content,
        page_number,
        section,
        source_file,
        metadata
    FROM multimodal_chunks
    WHERE chunk_type = 'text'
      AND to_tsvector('english', content)
          @@ plainto_tsquery('english', %(query)s)
    LIMIT %(k)s;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"query": query, "k": k})
            rows = cur.fetchall()

    return [
        Document(
            page_content=row["content"],
            metadata={
                "page": row.get("page_number"),
                "section": row.get("section"),
                "source": row.get("source_file"),
                "search_type": "fts",
            },
        )
        for row in rows
    ]