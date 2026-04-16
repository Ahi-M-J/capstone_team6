import os
from typing import TypedDict, List, Literal, Annotated, AsyncGenerator

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from src.tools.fts_search_tool import fts_search
from src.tools.vector_search_tool import query_documents
from src.tools.hybrid_search_tool import hybrid_search
from pydantic import BaseModel
from src.api.v1.schemas.query_schema import QueryResponse, RetrievedResult
from src.api.v1.schemas.query_schema import SubQueryResponse

from src.tools.tools import RAGState
from src.core.db import get_sql_database

load_dotenv(override=True)
os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1263111"


# ── Helpers ───────────────────────────────────────────────
def extract_text(output):
    """
    Normalizes ALL LLM + tool outputs into plain string
    """

    if output is None:
        return ""

    # LangChain AIMessage
    if hasattr(output, "content"):
        output = output.content

    # Cohere / tool list output
    if isinstance(output, list):
        texts = []
        for item in output:
            if isinstance(item, dict):
                texts.append(
                    item.get("text")
                    or item.get("content")
                    or str(item)
                )
            else:
                texts.append(str(item))
        return "\n".join(texts)

    if isinstance(output, dict):
        return output.get("text") or output.get("content") or str(output)

    return str(output)

def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )


def generate_hyde_query(query: str) -> str:
    print("\n[HyDE] Fallback triggered — generating hypothetical answer")
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Write a concise, factual answer to the user's question. "
         "This text will be used ONLY to improve document retrieval."),
        ("human", "{query}")
    ])
    response = (prompt | llm).invoke({"query": query})
    hyde_text = extract_text(response.content).strip()
    return hyde_text


def is_image_query(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in ["image", "picture", "figure", "diagram", "photo", "chart"])
# ── Query Splitter (NEW) ─────────────────────────



class SplitOutput(BaseModel):
    sub_queries: List[str]


def split_query(query: str) -> List[str]:
    llm = _get_llm()

    structured_llm = llm.with_structured_output(SplitOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """Split the user query into independent sub-queries.

Rules:
- If multiple questions exist → split them
- Keep each sub-query standalone
- If single query → return as-is

Example:
"What is RIL and what is total revenue?"
→ ["What is RIL?", "What is total revenue?"]
"""),
        ("human", "{query}")
    ])

    result = (prompt | structured_llm).invoke({"query": query})

    return result.sub_queries


# ── Streaming support ─────────────────────────────────────
async def stream_final_answer(context: str, query: str) -> AsyncGenerator[str, None]:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question using the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])
    chain = prompt | llm
    async for chunk in chain.astream({"context": context, "query": query}):
        if chunk.content:
            yield chunk.content


# ── Router Node ──────────────────────────────────────────
class _RouteDecision(BaseModel):
    route: Literal["product", "document"]
    reason: str


def router_node(state: RAGState) -> RAGState:
    query = state["query"].lower()

    # 🔥 HARD RULES (FAST + RELIABLE)
    sql_keywords = ["total", "sum", "average", "count", "max", "min"]

    if any(k in query for k in sql_keywords):
        print("[router_node] Forced → product (keyword match)")
        return {**state, "route": "product"}

    # else use LLM
    llm = _get_llm()
    structured_llm = llm.with_structured_output(_RouteDecision)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Route query:
- Use "document" for explanations, definitions, policies
- Use "product" ONLY if calculation is needed"""
        ),
        ("human", "Query: {query}")
    ])

    decision = (prompt | structured_llm).invoke({"query": state["query"]})

    return {**state, "route": decision.route}
# ── NL2SQL Node ──────────────────────────────────────────
def nl2sql_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    db = get_sql_database()

    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are a PostgreSQL expert.

STRICT RULES:
- Return ONLY SQL (no markdown, no explanation)
- Use ONLY provided schema
- No hallucinated columns
- Prefer card_transactions for spending queries
- LIMIT 50

SCHEMA:
{schema_info}
"""),
        ("human", "{query}")
    ])

    raw_sql = (sql_prompt | llm).invoke({"query": state["query"]})

    # ✅ FIX: ALWAYS STRING
    sql_query = extract_text(raw_sql).strip()

    # cleanup markdown
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

    try:
        sql_result = db.run(sql_query)
    except Exception as e:
        sql_result = f"SQL execution error: {e}"

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data analyst. Use ONLY SQL result."),
        ("human", "Question: {query}\nSQL: {sql}\nResult: {result}")
    ])

    raw_answer = (answer_prompt | llm).invoke({
        "query": state["query"],
        "sql": sql_query,
        "result": sql_result
    })

    final_answer = extract_text(raw_answer)

    return {
        **state,
        "sql_query": sql_query,
        "sql_result": sql_result,
        "final_answer": final_answer
    } 
# ── Tools Node ──────────────────────────────────────────
# ── Tools Node ──────────────────────────────────────────
@tool
def fts_search_tool(query: str, k: int = 20):
    """Perform full-text search on indexed documents and return top k results."""
    return fts_search(query, k)


@tool
def hybrid_search_tool(query: str, k: int = 20):
    """Perform hybrid search (vector + keyword) on documents and return top k results."""
    return hybrid_search(query, k)


@tool
def vector_search_tool(query: str, k: int = 20):
    """Perform semantic vector search on documents and return top k results."""
    return query_documents(query, k)


search_tools = [fts_search_tool, vector_search_tool, hybrid_search_tool]


def search_agent_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"),
                                 google_api_key=os.getenv("GOOGLE_API_KEY")).bind_tools(search_tools)
    response = llm.invoke([
        {"role": "system", "content": (
            "You are a retrieval agent. Choose the best search strategy:\n"
            "- Use `fts_search_tool` only if the query contains exact policy names, acronyms, specific document titles\n"
            "- Use `vector_search_tool` for semantic document retrieval\n"
            "- Use `hybrid_search_tool` if unsure or query is complex\nReturn only tool calls.")},
        {"role": "user", "content": state["query"]}
    ])
    return {**state, "messages": [response]}


# ── Search Result Node ─────────────────────────────────────
def search_result_node(state: RAGState) -> RAGState:
    docs = []
    for msg in state["messages"]:
        if isinstance(msg.content, list):
            docs.extend(msg.content)
    if docs:
        return {**state, "retrieved_docs": docs[:100]}
    hyde_query = generate_hyde_query(state["query"])
    hyde_docs = hybrid_search(hyde_query, k=25)
    return {**state, "hyde_query": hyde_query, "use_hyde": True, "retrieved_docs": hyde_docs[:100]}


# ── Rerank Node ──────────────────────────────────────────
def rerank_node(state: RAGState) -> RAGState:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    docs = state["retrieved_docs"][:50]
    if not docs:
        return {**state, "reranked_docs": []}
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=3
    )
    reranked_docs = [docs[r.index] for r in rerank_response.results]
    return {**state, "reranked_docs": reranked_docs}


# ── Decision Node ────────────────────────────────────────
def decision_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    context = "\n\n".join([doc.page_content for doc in state["reranked_docs"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return 'relevant' if the documents are relevant to the query, else 'not relevant'."),
        ("human", f"Question: {state['query']}\nContext: {context}")
    ])
    response = (prompt | llm).invoke({"query": state["query"], "context": context})
    label = extract_text(response.content).strip()
    if label == "relevant":
        return {**state, "response": "relevant"}
    elif label == "not relevant" and state["retries"] < 3:
        return {**state, "response": "retry"}
    return {**state, "response": "failed"}


# ── Generate Answer Node ──────────────────────────────────
def generate_answer_node(state: RAGState) -> RAGState:
    llm = _get_llm()

    context = "\n\n".join([
        doc.page_content for doc in state["reranked_docs"]
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from context"),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])

    answer = (prompt | llm).invoke({
        "context": context,
        "query": state["query"]
    })

    return {
        **state,
        "final_answer": extract_text(answer.content)
    }


# ── Rewrite Node ─────────────────────────────────────────
def rewrite_query_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's query in one concise sentence to improve document retrieval."),
        ("human", "{query}")
    ])
    rewritten = (prompt | llm).invoke({"query": state["query"]})
    return {**state, "query": extract_text(rewritten.content), "retries": state["retries"] + 1}


# ── Graph Construction ───────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("nl2sql", nl2sql_node)
    graph.add_node("search_agent", search_agent_node)
    tool_node = ToolNode(search_tools)
    graph.add_node("tools", tool_node)
    graph.add_node("search_result", search_result_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("decision", decision_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("rewrite", rewrite_query_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda s: s["route"], {"product": "nl2sql", "document": "search_agent"})
    graph.add_edge("nl2sql", END)
    graph.add_edge("search_agent", "tools")
    graph.add_edge("tools", "search_result")
    graph.add_edge("search_result", "rerank")
    graph.add_edge("rerank", "decision")
    graph.add_conditional_edges("decision", lambda s: s["response"], {"relevant": "generate_answer", "retry": "rewrite", "failed": "generate_answer"})
    graph.add_edge("rewrite", "search_agent")
    graph.add_edge("generate_answer", END)
    return graph.compile()
   
    compiled_agent = graph.compile()
    graph_image = compiled_agent.get_graph().draw_mermaid_png()
    with open("src/api/v1/agents/reranking_workflow.png", "wb") as f:
        f.write(graph_image)
            
    return compiled_agent



rag_graph = build_rag_graph()

def run_single_query(query: str) -> dict:
    initial_state: RAGState = {
        "query": query,
        "route": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "sql_query": "",
        "sql_result": "",
        "final_answer": "",
        "retries": 0,
        "messages": []
    }

    return rag_graph.invoke(initial_state)


# ── Public Entrypoint ───────────────────────────────────


def run_vector_search_agent(query: str) -> dict:
    sub_queries = split_query(query)

    all_responses = []

    for sub_q in sub_queries:
        state = run_single_query(sub_q)

        retrieved_results = []

        for i, doc in enumerate(state.get("reranked_docs", [])):
            retrieved_results.append(
                RetrievedResult(
                    chunk_id=i,
                    content=doc.page_content,
                    page=doc.metadata.get("page"),
                    section=doc.metadata.get("section"),
                    source=doc.metadata.get("source"),
                )
            )

        all_responses.append(
            SubQueryResponse(
                sub_query=sub_q,
                answer = str(state.get("final_answer", "")),

                retrieved_results=retrieved_results if state["route"] == "document" else [],

                sql_query=state.get("sql_query") if state["route"] == "product" else None,
                sql_result=state.get("sql_result") if state["route"] == "product" else None,
            )
        )

    final_response = QueryResponse(
        query=query,
        answers=all_responses
    )

    return final_response.model_dump()