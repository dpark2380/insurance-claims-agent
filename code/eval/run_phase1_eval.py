"""Run outputs/phase1_questions.json through retrieval + generation directly
(no shelling out to the CLI) and write outputs/phase1_eval.jsonl for manual review.

Run (from code/): python -m eval.run_phase1_eval
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from rag.generate import generate_answer  # noqa: E402
from rag.index import build_indexes  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402

QUESTIONS_PATH = ROOT / "outputs" / "phase1_questions.json"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
OUT_PATH = ROOT / "outputs" / "phase1_eval.jsonl"

CITATION_RE = re.compile(r"\[([^\[\]]+?,\s*page\s*\d+)\]", re.IGNORECASE)


def main():
    questions = json.loads(QUESTIONS_PATH.read_text())
    with CHUNKS_PATH.open() as f:
        chunks = [json.loads(line) for line in f]

    index_start = time.perf_counter()
    indexes = build_indexes(chunks)
    index_build_s = time.perf_counter() - index_start
    print(f"Index build (BM25+TF-IDF, {len(chunks)} chunks): {index_build_s:.3f}s")

    results = []
    retrieval_times, generation_times, total_times = [], [], []
    for item in questions:
        t0 = time.perf_counter()
        retrieved = retrieve(
            item["q"], chunks, indexes, top_k=5,
            insurer=item.get("insurer"), product=item.get("product"),
        )
        t1 = time.perf_counter()
        answer = generate_answer(item["q"], retrieved)
        t2 = time.perf_counter()

        retrieval_s, generation_s = t1 - t0, t2 - t1
        retrieval_times.append(retrieval_s)
        generation_times.append(generation_s)
        total_times.append(t2 - t0)

        results.append({
            "question": item["q"],
            "insurer": item.get("insurer"),
            "product": item.get("product"),
            "answer": answer,
            "retrieved_chunks": [
                {"insurer": c["insurer"], "product": c["product"], "doc_type": c["doc_type"], "page": c["page"]}
                for c in retrieved
            ],
            "citations_in_answer": CITATION_RE.findall(answer),
            "pass": None,  # fill in by hand after checking citations against the source PDF
            "retrieval_s": retrieval_s,
            "generation_s": generation_s,
        })
        print(f"[{item.get('insurer', 'ANY')}] {item['q'][:60]}... -> {len(retrieved)} chunks, "
              f"{len(CITATION_RE.findall(answer))} citations "
              f"(retrieval={retrieval_s:.3f}s, generation={generation_s:.3f}s)")

    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} results to {OUT_PATH}")
    print(f"Index build: {index_build_s:.3f}s")
    print(f"Retrieval  : median={statistics.median(retrieval_times):.3f}s  mean={statistics.mean(retrieval_times):.3f}s")
    print(f"Generation : median={statistics.median(generation_times):.3f}s  mean={statistics.mean(generation_times):.3f}s")
    print(f"End-to-end : median={statistics.median(total_times):.3f}s  mean={statistics.mean(total_times):.3f}s")


if __name__ == "__main__":
    main()
