from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str


class RetrievedResult(BaseModel):
    chunk_id: int
    content: str

    chunk_type: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    source: Optional[str] = None
    image_path: Optional[str] = None

    similarity: Optional[float] = None


class SubQueryResponse(BaseModel):
    sub_query: str
    answer: str

    retrieved_results: List[RetrievedResult] = []
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    answers: List[SubQueryResponse]