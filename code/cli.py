"""Phase 1 CLI: ask a coverage question, get a grounded answer with citations.

Run (from code/): python cli.py "is storm damage to my roof covered?" --insurer AAMI
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag.generate import generate_answer
from rag.index import build_indexes
from rag.retrieve import retrieve

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

load_dotenv(ROOT / ".env")


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open() as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--insurer")
    parser.add_argument("--product")
    parser.add_argument("--doc-type")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks()
    indexes = build_indexes(chunks)
    retrieved = retrieve(
        args.question, chunks, indexes, top_k=args.top_k,
        insurer=args.insurer, product=args.product, doc_type=args.doc_type,
    )

    if args.debug:
        print("--- retrieved chunks ---")
        for c in retrieved:
            print(f"[{c['insurer']} {c['product']} {c['doc_type']}, p{c['page']}] {c['text'][:120]}...")
        print("------------------------")

    answer = generate_answer(args.question, retrieved)
    print(answer)


if __name__ == "__main__":
    main()
