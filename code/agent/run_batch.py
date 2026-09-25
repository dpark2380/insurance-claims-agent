"""Phase 3 demo: run every claim in a JSONL file through the agent loop.

Run: python -m agent.run_batch data/claims/batch20.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.instrumentation import RECORDS, aggregate_report, set_current_claim, write_report

from extract.extract_fields import _get_model

from .loop import _get_index, run_claim

ROOT = Path(__file__).resolve().parents[2]
DECISIONS_DIR = ROOT / "outputs" / "decisions"
SUMMARY_PATH = ROOT / "outputs" / "decisions_summary.csv"
PHASE4_REPORT_PATH = ROOT / "outputs" / "phase4_report.md"

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
    parser.add_argument("--out-dir", type=Path, help="Write decisions/, summary and report here instead of outputs/ (e.g. for repeat runs).")
    args = parser.parse_args()

    decisions_dir, summary_path, report_path = DECISIONS_DIR, SUMMARY_PATH, PHASE4_REPORT_PATH
    if args.out_dir:
        decisions_dir = args.out_dir / "decisions"
        summary_path = args.out_dir / "decisions_summary.csv"
        report_path = args.out_dir / "phase4_report.md"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    with open(args.claims_path) as f:
        claims = [json.loads(line) for line in f]

    # Load the LoRA model and build the retrieval index up front so their
    # one-time cost is reported once, not folded into whichever claim
    # happens to trigger them first.
    start = time.monotonic()
    _get_model()
    _get_index()
    startup_s = time.monotonic() - start

    rows = []
    wall_times: dict[str, float] = {}
    for i, claim in enumerate(claims, start=1):
        # Fall back to a row number if a line is missing "claim_id" --
        # looking it up with claim["claim_id"] would raise KeyError *outside*
        # the try/except below and take down the whole batch, defeating the
        # point of isolating each claim's failure.
        claim_id = claim.get("claim_id", f"row-{i}")
        row = {"claim_id": claim_id, "decision": "", "escalated": "", "tool_calls_count": 0, "error": ""}
        set_current_claim(claim_id)
        start = time.monotonic()
        try:
            result = run_claim(claim["narrative"])
            result["claim_id"] = claim_id
            (decisions_dir / f"{claim_id}.json").write_text(json.dumps(result, indent=2))

            row["decision"] = result["final_decision"]
            row["escalated"] = result["final_decision"] == "escalate"
            row["tool_calls_count"] = sum(1 for t in result["turns"] if t["type"] == "tool_call")
        except Exception as e:
            row["error"] = str(e)
        finally:
            wall_times[claim_id] = time.monotonic() - start
            set_current_claim(None)
        rows.append(row)
        print(f"[{claim_id}] decision={row['decision'] or 'ERROR'} error={row['error']}")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "decision", "escalated", "tool_calls_count", "error"])
        writer.writeheader()
        writer.writerows(rows)

    n_errors = sum(1 for r in rows if r["error"])
    # row["escalated"] is only ever "" (init value, on error), True, or
    # False -- a plain truthy check is equivalent to `is True` here and
    # doesn't need the identity comparison.
    n_escalated = sum(1 for r in rows if r["escalated"])
    print(f"\n{len(rows)} claims processed, {n_errors} errors, {n_escalated} escalated")
    print(f"Summary: {summary_path}")

    write_report(RECORDS, report_path, wall_times, startup_s)
    agg = aggregate_report(RECORDS, wall_times)
    print(f"\nmean wall-clock/claim: {agg['mean_wall_s']:.2f}s  p95: {agg['p95_wall_s']:.2f}s  (startup {startup_s:.2f}s excluded)")
    print(f"mean API latency/claim: {agg['mean_latency_s']:.2f}s  p95: {agg['p95_latency_s']:.2f}s")
    print(f"extraction fallback calls: {agg['n_fallback_calls']} (${agg['fallback_cost_usd']:.4f})")
    print(f"mean cost/claim: ${agg['mean_cost_usd']:.4f}  total cost: ${agg['total_cost_usd']:.4f}")
    print(f"Phase 4 report: {report_path}")


if __name__ == "__main__":
    main()
