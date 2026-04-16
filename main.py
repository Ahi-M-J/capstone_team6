from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.routes import query, upload

app = FastAPI(title="Agentic Multimodal RAG")

# ✅ CORS (required for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ routers
app.include_router(query.router, prefix="/api/v1")
app.include_router(upload.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "RAG system running"}