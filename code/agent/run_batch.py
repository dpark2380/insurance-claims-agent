"""Phase 3 demo: run every claim in a JSONL file through the agent loop.

Run: python -m agent.run_batch data/claims/batch20.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from .loop import run_claim

ROOT = Path(__file__).resolve().parents[2]
DECISIONS_DIR = ROOT / "outputs" / "decisions"
SUMMARY_PATH = ROOT / "outputs" / "decisions_summary.csv"

load_dotenv(ROOT / ".env")


def main():
    """Run every claim in a JSONL file through `run_claim()` and summarize.

    Each claim is isolated in its own try/except -- one malformed row or one
    claim that makes the agent loop blow up must not stop the other 19 from
    being processed, so a per-claim failure is recorded as an `error` in the
    summary CSV instead of propagating.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("claims_path")
    args = parser.parse_args()

    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.claims_path) as f:
        claims = [json.loads(line) for line in f]

    rows = []
    for i, claim in enumerate(claims, start=1):
        # Fall back to a row number if a line is missing "claim_id" --
        # looking it up with claim["claim_id"] would raise KeyError *outside*
        # the try/except below and take down the whole batch, defeating the
        # point of isolating each claim's failure.
        claim_id = claim.get("claim_id", f"row-{i}")
        row = {"claim_id": claim_id, "decision": "", "escalated": "", "tool_calls_count": 0, "error": ""}
        try:
            result = run_claim(claim["narrative"])
            result["claim_id"] = claim_id
            (DECISIONS_DIR / f"{claim_id}.json").write_text(json.dumps(result, indent=2))

            row["decision"] = result["final_decision"]
            row["escalated"] = result["final_decision"] == "escalate"
            row["tool_calls_count"] = sum(1 for t in result["turns"] if t["type"] == "tool_call")
        except Exception as e:
            row["error"] = str(e)
        rows.append(row)
        print(f"[{claim_id}] decision={row['decision'] or 'ERROR'} error={row['error']}")

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "decision", "escalated", "tool_calls_count", "error"])
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if r["error"])
    # row["escalated"] is only ever "" (init value, on error), True, or
    # False -- a plain truthy check is equivalent to `is True` here and
    # doesn't need the identity comparison.
    n_escalated = sum(1 for r in rows if r["escalated"])
    print(f"\n{len(rows)} claims processed, {n_errors} errors, {n_escalated} escalated")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
