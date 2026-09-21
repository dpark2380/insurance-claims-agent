"""Phase 4, Step 4.1 -- cost/latency instrumentation for the full agent loop.

Wraps every `client.messages.create` call site (agent/loop.py's decision
loop, rag/generate.py's generate_answer, called via the retrieve_policy
tool) to record token usage and latency per call, tagged with whichever
claim is currently being processed, then aggregates per-claim.

Same $/MTok pricing already used and measured in extract/zero_shot.py.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

INPUT_PRICE = 2.00 / 1_000_000
OUTPUT_PRICE = 10.00 / 1_000_000

# Records accumulate here across an entire batch run; run_batch.py reads
# this at the end to build the report. Module-level, not per-instance,
# because the two instrumented call sites (loop.py, rag/generate.py) don't
# share an object to hang state off of.
RECORDS: list[dict] = []

_current_claim_id: str | None = None


def set_current_claim(claim_id: str | None) -> None:
    """Tag whichever claim is being processed right now, so calls made deep
    inside retrieve_policy (which doesn't itself know the claim_id) still
    get attributed correctly."""
    global _current_claim_id
    _current_claim_id = claim_id


def compute_cost(usage: dict) -> float:
    """usage -> $ cost. Cache tokens aren't used anywhere in this project's
    call sites (no cache_control on the agent loop or generate_answer's
    system prompt beyond what's already there), so only input/output need
    pricing -- if cache fields are added later this needs a cache-read rate
    too, since cache reads are billed well below the fresh-input rate."""
    return usage.get("input_tokens", 0) * INPUT_PRICE + usage.get("output_tokens", 0) * OUTPUT_PRICE


@contextmanager
def instrument_call(call_site: str):
    """Wrap one `messages.create(...)` call. Usage:

        with instrument_call("loop") as rec:
            response = client.messages.create(...)
            rec["response"] = response

    Records latency regardless of whether the call raises; only records
    token usage if the call actually returned a response.
    """
    rec = {"claim_id": _current_claim_id, "call_site": call_site, "response": None}
    start = time.monotonic()
    try:
        yield rec
    finally:
        elapsed = time.monotonic() - start
        response = rec.pop("response", None)
        if response is not None:
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            rec.update(usage)
            rec["cost_usd"] = compute_cost(usage)
        else:
            rec["input_tokens"] = rec["output_tokens"] = rec["cost_usd"] = 0
        rec["latency_s"] = elapsed
        RECORDS.append(rec)


def aggregate_report(records: list[dict]) -> dict:
    """Per-claim totals (sum of every instrumented call tagged to that
    claim), then mean/p95 across claims. Records with no claim_id (e.g. a
    stray Phase 1 CLI call made outside a batch run) are excluded from the
    per-claim breakdown -- there's no claim to attribute them to."""
    by_claim: dict[str, list[dict]] = {}
    for r in records:
        if r["claim_id"] is None:
            continue
        by_claim.setdefault(r["claim_id"], []).append(r)

    per_claim = {}
    for claim_id, calls in by_claim.items():
        per_claim[claim_id] = {
            "n_calls": len(calls),
            "total_latency_s": sum(c["latency_s"] for c in calls),
            "total_cost_usd": sum(c["cost_usd"] for c in calls),
            "total_input_tokens": sum(c["input_tokens"] for c in calls),
            "total_output_tokens": sum(c["output_tokens"] for c in calls),
        }

    latencies = [v["total_latency_s"] for v in per_claim.values()]
    costs = [v["total_cost_usd"] for v in per_claim.values()]

    return {
        "n_claims": len(per_claim),
        "per_claim": per_claim,
        "mean_latency_s": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency_s": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else (latencies[0] if latencies else 0.0),
        "mean_cost_usd": statistics.mean(costs) if costs else 0.0,
        "total_cost_usd": sum(costs),
    }


def write_report(records: list[dict], out_path) -> None:
    agg = aggregate_report(records)
    lines = [
        "# Phase 4 cost/latency report",
        "",
        f"n claims: {agg['n_claims']}",
        f"mean latency/claim: {agg['mean_latency_s']:.2f}s",
        f"p95 latency/claim: {agg['p95_latency_s']:.2f}s",
        f"mean cost/claim: ${agg['mean_cost_usd']:.4f}",
        f"total cost: ${agg['total_cost_usd']:.4f}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # Smoke test: compute_cost on a hand-built usage dict with a known
    # expected cost, and aggregate_report on synthetic per-call records.
    cost = compute_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - 12.00) < 1e-9, f"expected $12.00 for 1M in + 1M out, got {cost}"

    fake_records = [
        {"claim_id": "c1", "call_site": "loop", "input_tokens": 1000, "output_tokens": 100, "cost_usd": compute_cost({"input_tokens": 1000, "output_tokens": 100}), "latency_s": 1.0},
        {"claim_id": "c1", "call_site": "retrieve", "input_tokens": 500, "output_tokens": 50, "cost_usd": compute_cost({"input_tokens": 500, "output_tokens": 50}), "latency_s": 2.0},
        {"claim_id": "c2", "call_site": "loop", "input_tokens": 1000, "output_tokens": 100, "cost_usd": compute_cost({"input_tokens": 1000, "output_tokens": 100}), "latency_s": 1.5},
        {"claim_id": None, "call_site": "cli", "input_tokens": 200, "output_tokens": 20, "cost_usd": compute_cost({"input_tokens": 200, "output_tokens": 20}), "latency_s": 0.5},
    ]
    agg = aggregate_report(fake_records)
    assert agg["n_claims"] == 2, agg
    assert abs(agg["per_claim"]["c1"]["total_latency_s"] - 3.0) < 1e-9
    assert abs(agg["per_claim"]["c2"]["total_latency_s"] - 1.5) < 1e-9
    print(f"cost smoke test: {cost}")
    print(f"aggregate smoke test: {agg}")
    print("\nsmoke tests passed")
