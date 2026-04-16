from typing import TypedDict, List
from langchain_core.documents import Document
from typing import Annotated
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    query: str
    route: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    sql_query: str
    sql_result: str
    final_answer: str
    retries: int
    messages: list