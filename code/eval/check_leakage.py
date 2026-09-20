"""Check for near-duplicate narratives leaking across train/val/test splits.

All three splits are drawn from the same synthetic generation batch/prompt,
so nothing guarantees a paraphrase of a train example didn't land in test --
that would silently inflate eval scores. Reuses the project's own TF-IDF
cosine similarity (rag/index.py) rather than pulling in a new embedding
dependency; this is a lexical-overlap check, not semantic, but a near-exact
paraphrase leak still shows up as high cosine similarity on shared vocabulary.

Run: python -m eval.check_leakage
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.index import build_vocabulary, compute_idf, cosine_similarity, tfidf_embed

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_DIR = ROOT / "data" / "claims"

# Above this, two narratives are similar enough to be treated as a
# near-duplicate rather than two distinct claims that happen to share
# insurance-domain vocabulary (which is expected and not a leak).
THRESHOLD = 0.85


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def check_pair(name_a: str, rows_a: list[dict], name_b: str, rows_b: list[dict], vocab: dict, idf: dict) -> list[tuple[int, int, float]]:
    emb_a = [tfidf_embed(r["narrative"], vocab, idf) for r in rows_a]
    emb_b = [tfidf_embed(r["narrative"], vocab, idf) for r in rows_b]

    flagged = []
    max_sim = 0.0
    for i, ea in enumerate(emb_a):
        for j, eb in enumerate(emb_b):
            sim = cosine_similarity(ea, eb)
            max_sim = max(max_sim, sim)
            if sim >= THRESHOLD:
                flagged.append((i, j, sim))

    print(f"{name_a} vs {name_b}: {len(flagged)} pairs >= {THRESHOLD} (max similarity found: {max_sim:.3f})")
    for i, j, sim in flagged[:5]:
        print(f"  sim={sim:.3f}")
        print(f"    {name_a}[{i}]: {rows_a[i]['narrative'][:100]!r}")
        print(f"    {name_b}[{j}]: {rows_b[j]['narrative'][:100]!r}")
    return flagged


def main():
    train = load(CLAIMS_DIR / "train.jsonl")
    val = load(CLAIMS_DIR / "val.jsonl")
    test = load(CLAIMS_DIR / "test.jsonl")

    # Shared vocabulary/IDF across all three so similarity scores are
    # comparable across the three pairwise checks below.
    all_narratives = [r["narrative"] for r in train + val + test]
    vocab = build_vocabulary(all_narratives)
    idf = compute_idf(all_narratives, vocab)

    total_flagged = 0
    total_flagged += len(check_pair("train", train, "test", test, vocab, idf))
    total_flagged += len(check_pair("train", train, "val", val, vocab, idf))
    total_flagged += len(check_pair("val", val, "test", test, vocab, idf))

    print(f"\n{total_flagged} total near-duplicate pairs found across all split boundaries")


if __name__ == "__main__":
    main()
