"""Chunk extracted page records into overlapping word windows, carrying metadata."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_page_records(records: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """records: list of per-page dicts from extract_text.py. Returns list of
    chunk dicts with a stable chunk_id and the source page/doc metadata."""
    out = []
    for rec in records:
        if rec["low_content"] or not rec["text"]:
            continue
        for i, chunk in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            out.append({
                "chunk_id": f"{rec['doc_id']}-p{rec['page']}-c{i}",
                "doc_id": rec["doc_id"],
                "insurer": rec["insurer"],
                "product": rec["product"],
                "doc_type": rec["doc_type"],
                "page": rec["page"],
                "text": chunk,
            })
    return out
