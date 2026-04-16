# Credit Card Spend Analyzer (Agentic RAG)

This project is an intelligent credit card statement analyzer that allows users to query their spending data using natural language. It uses an Agentic Retrieval-Augmented Generation (RAG) approach to retrieve and generate accurate responses from transaction data.


## Features

Natural language querying over credit card transactions
Agent-based decision making for retrieval strategy
Hybrid search (Full-Text Search + Vector Search)
Query rephrasing for better results
Reranking using Cohere
Supports ingestion of statement data (PDF/CSV)
Modular and scalable architecture

## Tech Stack

Backend: FastAPI
Database: PostgreSQL with pgvector
LLM: Google Gemini
Reranking: Cohere
Orchestration: LangGraph
Embeddings: Google Generative AI


## Project Structure
src/
├── api/v1/
│   ├── routes/
│   │   ├── query.py
│   │   └── upload.py
│   ├── schemas/
│   │   └── query_schema.py
│   ├── services/
│   │   └── query_service.py
│   └── agents/
│       └── agent.py
│
├── tools/
│   ├── fts_search_tool.py
│   ├── vector_search_tool.py
│   ├── hybrid_search_tool.py
│   └── tool.py
│
├── core/
│   └── db.py
│
├── ingestion/
│   └── ingestion.py
│
└── main.py


## Setup Instructions

### 1. Clone the repository
git clone <your-repo-url>
cd <project-folder>


### 2. Create virtual environment
uv venv
.venv\Scripts\activate   # Windows

---

### 3. Install dependencies
uv pip install -r requirements.txt

---

### 4. Setup PostgreSQL

Make sure PostgreSQL is running and create database:
CREATE DATABASE ragdb;

Enable pgvector:
CREATE EXTENSION vector;

---

### 5. Configure environment variables

Create a .env file in root:
PG_CONNECTION=postgresql+psycopg://postgres:password@localhost:5432/ragdb
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key

---

### 6. Run ingestion
python -m src.ingestion.ingestion

---

### 7. Run the application
uvicorn main:app --reload

---

## API Endpoints

### Query
POST /api/v1/query

Request:
{
  "question": "What is my total spend on dining?"
}

---

### Upload
POST /api/v1/upload

Upload credit card statement files.

---

## How It Works

1. User sends a query
2. Agent decides which search tool to use
3. Relevant data is retrieved from database
4. Results are reranked
5. LLM generates final answer

---

## Example Queries

What is my total spend this month?
How much did I spend on groceries?
Show all transactions above ₹5000
Which category has the highest spending?


## Future Enhancements

Frontend dashboard
Streaming responses
NL2SQL integration
Advanced analytics and insights
Alerts and recommendations


