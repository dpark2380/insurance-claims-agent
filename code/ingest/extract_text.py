"""Extract per-page text from every PDF in data/manifest.json.

Run (from code/): python -m ingest.extract_text
"""

from __future__ import annotations

import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
OUT_DIR = ROOT / "data" / "extracted"

MIN_WORDS = 40  # pages shorter than this are usually covers/TOC -- flag, don't drop


def extract_doc(doc_id: str, meta: dict) -> list[dict]:
    pdf_path = ROOT / meta["path"]
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            records.append({
                "doc_id": doc_id,
                "insurer": meta["insurer"],
                "product": meta["product"],
                "doc_type": meta["doc_type"],
                "page": page_num,
                "text": text.strip(),
                "low_content": len(text.split()) < MIN_WORDS,
            })
    return records


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_pages = 0
    total_low = 0
    for doc_id, meta in manifest.items():
        records = extract_doc(doc_id, meta)
        out_path = OUT_DIR / f"{doc_id}.jsonl"
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        n_low = sum(r["low_content"] for r in records)
        total_pages += len(records)
        total_low += n_low
        print(f"[{doc_id}] {len(records)} pages -> {out_path.name} ({n_low} low-content)")
    print(f"\n{len(manifest)} docs, {total_pages} pages total, {total_low} low-content")


if __name__ == "__main__":
    main()
