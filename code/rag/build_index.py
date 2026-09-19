"""Build data/chunks.jsonl from the per-page JSONL files in data/extracted/.

Run (from code/): python -m rag.build_index
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .chunking import chunk_page_records

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = ROOT / "data" / "extracted"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"


def load_all_page_records() -> list[dict]:
    records = []
    for path in sorted(EXTRACTED_DIR.glob("*.jsonl")):
        with path.open() as f:
            records.extend(json.loads(line) for line in f)
    return records


def main():
    start = time.perf_counter()
    records = load_all_page_records()
    chunks = chunk_page_records(records)
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    elapsed = time.perf_counter() - start
    print(f"{len(records)} pages -> {len(chunks)} chunks -> {CHUNKS_PATH} ({elapsed:.3f}s)")


if __name__ == "__main__":
    main()
