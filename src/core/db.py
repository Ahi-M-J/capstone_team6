import os
import base64
import hashlib
import json
import pathlib
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row


load_dotenv()

_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

_EMBED_BATCH_SIZE = 50



_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDINGS_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536
)


def get_vector_store(collection_name: str = "hr_support_desk") -> PGVector:
    return PGVector(
        collection_name=collection_name,
        connection=_PG_CONNECTION,
        embeddings=_embeddings_model,
        use_jsonb=True,
    )


def get_sql_database() -> SQLDatabase:
    """Return a LangChain SQLDatabase connected to the agentic_rag_db (read-only).

    Uses the rag_readonly role from sql/seed.sql — SELECT privileges only.
    Connection string is read from AGENTIC_RAG_DB_URL in the environment.
    """
    db_url = os.getenv("AGENTIC_RAG_DB_URL")
    if not db_url:
        raise ValueError("AGENTIC_RAG_DB_URL is not set. Check your .env file.")
    return SQLDatabase.from_uri(

        db_url,
        include_tables=[
            "customers",
            "credit_cards",
            "card_transactions",
            "reward_transactions",
            "billing_statements",
        ],
        sample_rows_in_table_info=5,
        view_support=True,
    )

# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the existing doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0

    contents = [c["content"] for c in chunks]

    # ── Batch embed ─────────────────────────────────────────────
    all_embeddings = []

    for i, text in enumerate(contents):
        print(f"[DEBUG] embedding {i+1}/{len(contents)}")
        emb = _embeddings_model.embed_query(text)  # 👈 IMPORTANT CHANGE
        all_embeddings.append(emb)
        

    # 🚨 Safety check
    if len(all_embeddings) != len(chunks):
        raise ValueError("Mismatch between chunks and embeddings")

    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0

    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:

                # 🧹 साफ previous chunks
                cur.execute(
                    "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                    (doc_id,),
                )

                for idx, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                    meta = chunk["metadata"]

                    # ── Image handling ─────────────────────────────
                    img_b64 = meta.get("image_base64")
                    image_path: str | None = None
                    mime_type = None

                    if img_b64:
                        image_bytes = base64.b64decode(img_b64)
                        img_dir = pathlib.Path("data/images")
                        img_dir.mkdir(parents=True, exist_ok=True)

                        img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                        img_file = img_dir / f"{doc_id}_{img_hash}.png"

                        img_file.write_bytes(image_bytes)
                        image_path = str(img_file)
                        mime_type = "image/png"

                    # ── Embedding format ──────────────────────────
                    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                    # ── Clean metadata ────────────────────────────
                    clean_meta = {
                        k: v for k, v in meta.items()
                        if k not in _DEDICATED_COLUMNS
                    }

                    # 🔍 Debug each insert
                    print(f"[DEBUG] inserting chunk {idx+1}")

                    cur.execute(
                        """
                        INSERT INTO multimodal_chunks (
                            doc_id, chunk_type, element_type, content,
                            image_path, mime_type,
                            page_number, section, source_file,
                            position, embedding, metadata
                        ) VALUES (
                            %s::uuid, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s,
                            %s::jsonb, %s::vector, %s::jsonb
                        )
                        """,
                        (
                            doc_id,
                            chunk["content_type"],
                            meta.get("element_type"),
                            chunk["content"],
                            image_path,  # ✅ FIXED
                            mime_type,
                            meta.get("page_number"),
                            meta.get("section"),
                            meta.get("source_file"),
                            json.dumps(meta.get("position")) if meta.get("position") else None,
                            embedding_str,
                            json.dumps(clean_meta),
                        ),
                    )

                    rows_inserted += 1

            conn.commit()

    except Exception as e:
        print("❌ ERROR inserting chunks:", e)
        raise

    print(f"[DEBUG] inserted total: {rows_inserted}")

    return rows_inserted
# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def similarity_search(
    query: str,
    k: int = 5,
    chunk_type: str | None = None,
) -> list[dict]:

    query_vec = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "vec": embedding_str,
                "chunk_type": chunk_type,
                "k": k
            })
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)

        img_path = row.get("image_path")

        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                row["image_base64"] = base64.b64encode(f.read()).decode()
        else:
            row["image_base64"] = None

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type,mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results
