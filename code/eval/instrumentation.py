"""Phase 4, Step 4.1 -- cost/latency instrumentation for the full agent loop.

Wraps every `client.messages.create` call site (agent/loop.py's decision
loop, rag/generate.py's generate_answer via the retrieve_policy tool, and
extract/zero_shot.py's extraction fallback) to record token usage and
latency per call, tagged with whichever claim is currently being processed,
then aggregates per-claim. run_batch.py adds each claim's wall-clock time,
which also covers what API timing can't see: local LoRA inference,
retrieval, and tool execution.

Same $/MTok pricing already used and measured in extract/zero_shot.py.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager

INPUT_PRICE = 2.00 / 1_000_000
OUTPUT_PRICE = 10.00 / 1_000_000
# Prompt caching on claude-sonnet-5: 5-minute-TTL writes bill at 1.25x input,
# reads at 0.1x. The API's `input_tokens` excludes both, so they're priced here
# separately rather than silently dropped.
CACHE_WRITE_PRICE = INPUT_PRICE * 1.25
CACHE_READ_PRICE = INPUT_PRICE * 0.10

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
    """usage -> $ cost, including cache writes/reads (absent keys count as 0)."""
    return (usage.get("input_tokens", 0) * INPUT_PRICE + usage.get("output_tokens", 0) * OUTPUT_PRICE
            + usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_PRICE
            + usage.get("cache_read_input_tokens", 0) * CACHE_READ_PRICE)


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
            u = response.usage
            usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                     "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                     "cache_read_input_tokens": u.cache_read_input_tokens or 0}
            rec.update(usage)
            rec["cost_usd"] = compute_cost(usage)
        else:
            rec["input_tokens"] = rec["output_tokens"] = rec["cost_usd"] = 0
            rec["cache_creation_input_tokens"] = rec["cache_read_input_tokens"] = 0
        rec["latency_s"] = elapsed
        RECORDS.append(rec)


def _mean_p95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 2 else values[0]
    return statistics.mean(values), p95


def aggregate_report(records: list[dict], wall_times: dict[str, float] | None = None) -> dict:
    """Per-claim totals (sum of every instrumented call tagged to that
    claim), then mean/p95 across claims. Records with no claim_id (e.g. a
    stray Phase 1 CLI call made outside a batch run) are excluded from the
    per-claim breakdown -- there's no claim to attribute them to.
    `wall_times` (claim_id -> seconds) adds end-to-end latency stats."""
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

    costs = [v["total_cost_usd"] for v in per_claim.values()]
    mean_api, p95_api = _mean_p95([v["total_latency_s"] for v in per_claim.values()])
    wall = list((wall_times or {}).values())
    mean_wall, p95_wall = _mean_p95(wall)
    tagged = [r for r in records if r["claim_id"] is not None]

    return {
        "n_claims": len(per_claim),
        "per_claim": per_claim,
        "mean_latency_s": mean_api,
        "p95_latency_s": p95_api,
        "n_wall_claims": len(wall),
        "mean_wall_s": mean_wall,
        "p95_wall_s": p95_wall,
        "mean_cost_usd": statistics.mean(costs) if costs else 0.0,
        "total_cost_usd": sum(costs),
        "n_fallback_calls": sum(1 for r in tagged if r["call_site"] == "extract_fallback"),
        "fallback_cost_usd": sum(r["cost_usd"] for r in tagged if r["call_site"] == "extract_fallback"),
        "cache_write_tokens": sum(r.get("cache_creation_input_tokens", 0) for r in tagged),
        "cache_read_tokens": sum(r.get("cache_read_input_tokens", 0) for r in tagged),
    }


def write_report(records: list[dict], out_path, wall_times: dict[str, float] | None = None, startup_s: float | None = None) -> None:
    agg = aggregate_report(records, wall_times)
    lines = [
        "# Phase 4 cost/latency report",
        "",
        f"n claims: {agg['n_claims']}",
        f"mean wall-clock latency/claim (end to end): {agg['mean_wall_s']:.2f}s",
        f"p95 wall-clock latency/claim (end to end): {agg['p95_wall_s']:.2f}s",
        f"mean API latency/claim: {agg['mean_latency_s']:.2f}s",
        f"p95 API latency/claim: {agg['p95_latency_s']:.2f}s",
        f"mean cost/claim (incl. extraction fallback): ${agg['mean_cost_usd']:.4f}",
        f"total cost: ${agg['total_cost_usd']:.4f}",
        f"extraction fallback calls: {agg['n_fallback_calls']} (${agg['fallback_cost_usd']:.4f})",
        f"prompt cache tokens: {agg['cache_write_tokens']} written, {agg['cache_read_tokens']} read"
        + (f" (read:write {agg['cache_read_tokens'] / agg['cache_write_tokens']:.2f})" if agg["cache_write_tokens"]
           else " (no cache writes: cached prefix is under the model's minimum cacheable length or unmarked)"),
    ]
    if startup_s is not None:
        lines.append(f"one-time startup (LoRA load + index build, excluded above): {startup_s:.2f}s")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # Smoke test: compute_cost on a hand-built usage dict with a known
    # expected cost, and aggregate_report on synthetic per-call records.
    cost = compute_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert abs(cost - 12.00) < 1e-9, f"expected $12.00 for 1M in + 1M out, got {cost}"
    cache_cost = compute_cost({"cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000})
    assert abs(cache_cost - (2.50 + 0.20)) < 1e-9, cache_cost

    fake_records = [
        {"claim_id": "c1", "call_site": "loop", "input_tokens": 1000, "output_tokens": 100, "cost_usd": compute_cost({"input_tokens": 1000, "output_tokens": 100}), "latency_s": 1.0},
        {"claim_id": "c1", "call_site": "retrieve", "input_tokens": 500, "output_tokens": 50, "cost_usd": compute_cost({"input_tokens": 500, "output_tokens": 50}), "latency_s": 2.0},
        {"claim_id": "c2", "call_site": "loop", "input_tokens": 1000, "output_tokens": 100, "cost_usd": compute_cost({"input_tokens": 1000, "output_tokens": 100}), "latency_s": 1.5},
        {"claim_id": None, "call_site": "cli", "input_tokens": 200, "output_tokens": 20, "cost_usd": compute_cost({"input_tokens": 200, "output_tokens": 20}), "latency_s": 0.5},
    ]
    fake_records.append({"claim_id": "c2", "call_site": "extract_fallback", "input_tokens": 100, "output_tokens": 10, "cost_usd": 0.5, "latency_s": 0.1})
    agg = aggregate_report(fake_records, {"c1": 4.0, "c2": 2.0})
    assert agg["n_claims"] == 2, agg
    assert agg["n_fallback_calls"] == 1 and agg["fallback_cost_usd"] == 0.5, agg
    assert agg["mean_wall_s"] == 3.0, agg
    assert abs(agg["per_claim"]["c1"]["total_latency_s"] - 3.0) < 1e-9
    assert abs(agg["per_claim"]["c2"]["total_latency_s"] - 1.6) < 1e-9
    print(f"cost smoke test: {cost}")
    print(f"aggregate smoke test: {agg}")
    print("\nsmoke tests passed")
