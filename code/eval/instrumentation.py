"""Phase 4, Step 4.1 -- cost/latency instrumentation. SCAFFOLD ONLY, not implemented.

Goal: wrap every `client.messages.create` call site (retrieve_policy,
extract_fields's Claude fallback, the agent loop) to record token usage and
latency, then aggregate a batch's worth into outputs/phase4_report.md.
"""

from __future__ import annotations

# TODO: hardcode a pricing table for the model(s) in use (see claude-api
# skill / current Anthropic pricing page for $/MTok input vs output).
PRICING = {}


def instrument_call(fn):
    """TODO: decorator (or context manager) around a `messages.create` call.

    Should capture, per call: input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, and wall-clock
    latency (time.monotonic() around the call). Needs a place to stash these
    records so a batch run can aggregate them afterward -- decide whether
    that's a module-level list, a return-value wrapper, or something written
    straight to disk per call.
    """
    raise NotImplementedError


def compute_cost(usage: dict) -> float:
    """TODO: usage dict -> $ cost using PRICING. Remember cache-read tokens
    are priced differently (cheaper) than fresh input tokens -- don't just
    price everything at the input-token rate."""
    raise NotImplementedError


def aggregate_report(records: list[dict]) -> dict:
    """TODO: turn a list of per-call records into the report.md numbers --
    total $, $/claim, p50/p95 latency, cache read:write ratio."""
    raise NotImplementedError


def write_report(records: list[dict], out_path) -> None:
    """TODO: render aggregate_report()'s numbers into outputs/phase4_report.md."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: smoke test once implemented, matching this project's convention
    # (see extract/eval.py, agent/tools.py) -- e.g. compute_cost() on a
    # couple of hand-built usage dicts with known expected costs.
    pass
