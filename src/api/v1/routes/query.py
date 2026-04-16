from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from src.api.v1.services.query_service import query_documents

router = APIRouter()


class QueryRequest(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None


@router.post("/query")
def query_endpoint(request: QueryRequest):
    user_query = request.question or request.query

    if not user_query:
        return {"error": "No query provided"}

    return query_documents(user_query)