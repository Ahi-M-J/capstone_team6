import os
import pathlib
from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document

load_dotenv()

_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:  # ❗ remove empty chunks
            chunks.append(chunk)
        start += step

    return chunks


def run_ingestion(file_path: str) -> dict:
    resolved = pathlib.Path(file_path).resolve()

    # Step 1: document register
    doc_id = upsert_document(resolved.name, str(resolved))
    print(f"[ingestion] doc_id={doc_id} file={file_path}")

    # Step 2: parse
    print(f"[ingestion] Parsing: {file_path}")
    parsed_elements = parse_document(file_path)
    print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

    chunks = []

    for elem in parsed_elements:
        content = elem.get("content", "")

        if not content or not content.strip():
            continue  # ❗ SKIP EMPTY CONTENT

        content_type = elem.get("content_type", "text")
        metadata = elem.get("metadata", {}) or {}
        image_path = elem.get("image_path")

        # TEXT chunking
        if content_type == "text" and len(content) > _TEXT_CHUNK_SIZE:
            sub_chunks = _split_text(
                content,
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP
            )

            for sub in sub_chunks:
                if not sub.strip():
                    continue

                chunks.append({
                    "content": sub,
                    "content_type": content_type,
                    "chunk_type": content_type,
                    "image_path": image_path,
                    "metadata": metadata,
                })

        else:
            chunks.append({
                "content": content,
                "content_type": content_type,
                "chunk_type": content_type,
                "image_path": image_path,
                "metadata": metadata,
            })

    # ❗ FINAL CLEAN CHECK
    chunks = [c for c in chunks if c.get("content") and c["content"].strip()]

    print(f"[ingestion] {len(chunks)} valid chunks ready")

    if not chunks:
        raise ValueError("No valid chunks generated from document")

    # Step 4: store
    count = store_chunks(chunks, doc_id)

    print(f"[ingestion] Stored {count} chunks → DB")

    return {
        "status": "success",
        "doc_id": doc_id,
        "chunks_ingested": count
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        pdf_path = pathlib.Path(sys.argv[1])
    else:
        pdf_path = pathlib.Path("data/sample.pdf")

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    result = run_ingestion(str(pdf_path))
    print(result)
