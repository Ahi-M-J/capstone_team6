# src/api/v1/tools/vector_search_tool.py
from typing import List
from langchain_core.documents import Document
from src.core.db import similarity_search


def query_documents(query: str, k: int = 25) -> List[Document]:
    rows = similarity_search(query=query, k=k, chunk_type="text")

    return [
        Document(
            page_content=row["content"],
            metadata={
                "page": row.get("page_number"),
                "section": row.get("section"),
                "source": row.get("source_file"),
                "chunk_type": row.get("chunk_type"),
                "similarity": row.get("similarity"),
                "search_type": "vector",
            },
        )
        for row in rows
    ]