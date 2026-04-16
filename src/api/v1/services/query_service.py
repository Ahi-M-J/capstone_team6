from src.api.v1.agents.agent import run_vector_search_agent

def query_documents(query: str):
    return run_vector_search_agent(query)