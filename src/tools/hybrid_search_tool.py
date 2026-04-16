# src/api/v1/tools/hybrid_search_tool.py
from typing import List, Dict
from langchain_core.documents import Document
from src.tools.vector_search_tool import query_documents
from src.tools.fts_search_tool import fts_search


def hybrid_search(query: str, k: int = 25) -> List[Document]:
    vector_docs = query_documents(query, k=k)
    fts_docs = fts_search(query, k=k)

    RRF_K = 60
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}

    def key(doc: Document) -> str:
        return doc.page_content[:200]

    for rank, doc in enumerate(vector_docs):
        score = 1 / (RRF_K + rank + 1)
        scores[key(doc)] = scores.get(key(doc), 0) + score
        docs[key(doc)] = doc

    for rank, doc in enumerate(fts_docs):
        score = 1 / (RRF_K + rank + 1)
        scores[key(doc)] = scores.get(key(doc), 0) + score
        docs.setdefault(key(doc), doc)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[k] for k, _ in ranked[:k]]