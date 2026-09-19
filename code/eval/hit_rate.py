"""Retrieval-quality benchmark: hit-rate@k and MRR@k for Phase 1 RAG.

Ground truth (data/phase1_ground_truth.json) was derived from
outputs/phase1_eval.jsonl's human-verified citations (the `pass` field was
set by checking each citation against the real source PDF text -- see
PROJECT_LOG.md, "Step 1.7 verification"). One question (RealInsurance
malicious-damage) failed generation but its correct page was independently
located by grepping the source text for the exclusion clause. One question
(NRMA satellite dish / space tourism) is an adversarial no-answer probe --
no real policy page addresses it, so it's excluded from ground truth
entirely rather than forced to have a "correct" chunk.

Run (from code/): python -m eval.hit_rate
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "outputs" / "phase1_eval.jsonl"
GROUND_TRUTH_PATH = ROOT / "outputs" / "phase1_ground_truth.json"


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    rows = [json.loads(line) for line in EVAL_PATH.read_text().splitlines() if line.strip()]
    by_question = {r["question"]: r for r in rows}

    hits, reciprocal_ranks, skipped = [], [], []
    for gt in ground_truth:
        row = by_question[gt["question"]]
        if gt.get("no_ground_truth"):
            skipped.append(gt["question"])
            continue
        relevant = {(p["doc_type"], p["page"]) for p in gt["relevant_pages"]}

        rank = None
        for i, chunk in enumerate(row["retrieved_chunks"], start=1):
            if (chunk["doc_type"], chunk["page"]) in relevant:
                rank = i
                break

        hits.append(1 if rank is not None else 0)
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
        print(f"{'HIT ' if rank else 'MISS'} rank={rank}  {gt['question'][:60]}")

    n = len(hits)
    print(f"\nn={n} (excluded {len(skipped)} no-ground-truth question(s): {skipped})")
    print(f"hit-rate@5 = {sum(hits)}/{n} = {sum(hits) / n:.3f}")
    print(f"MRR@5      = {statistics.mean(reciprocal_ranks):.3f}")


if __name__ == "__main__":
    main()
