"""Tool-calling agent loop: claim narrative -> approve/deny/escalate decision.

`run_claim()` is the only public entry point -- run_batch.py and any future
Phase 4/5 instrumentation call this, not the internals.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.generate import MODEL, _client
from rag.index import build_indexes

from .tools import (
    TOOLS,
    run_calculate_payout,
    run_escalate_to_human,
    run_extract_fields,
    run_get_policy_certificate,
    run_retrieve_policy,
)

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

MAX_TURNS = 8

SYSTEM_PROMPT = (
    "You are an insurance claims triage agent for Australian home insurance. "
    "For every claim: call extract_fields first, then use retrieve_policy to "
    "check whether the claim's cause is covered by the named insurer's "
    "policy text. If the claim names a specific insurer, also call "
    "get_policy_certificate for that insurer's excess/coverage_limit, then "
    "call calculate_payout with those real numbers, then state a final "
    "decision.\n\n"
    "Escalation policy -- call escalate_to_human instead of deciding when ANY "
    "of these are true:\n"
    "- retrieve_policy returns contradictory or ambiguous coverage text\n"
    "- extract_fields returns a null estimated_amount you cannot resolve "
    "another way\n"
    "- the claimed cause directly conflicts with a policy exclusion you found\n"
    "- the claim does not name a specific insurer, so get_policy_certificate "
    "cannot be resolved and calculate_payout would need a guessed number\n\n"
    "Never guess a coverage answer, an excess amount, or a coverage limit -- "
    "only use numbers retrieve_policy or get_policy_certificate actually "
    "returned. When you are done, "
    "reply with a final message (no more tool calls) starting with exactly "
    "one of 'DECISION: approve', 'DECISION: deny', or 'DECISION: escalate', "
    "followed by your reasoning and any citations you relied on."
)

_chunks = None
_indexes = None


def _get_index():
    """Lazily build (once per process) and cache the retrieval index.

    `chunks.jsonl` + BM25/TF-IDF indexing only need to happen once no matter
    how many claims `run_claim()` processes in a batch -- rebuilding per
    claim would be pure wasted work, so the result is memoized in module
    globals the first time `retrieve_policy` is actually called.
    """
    global _chunks, _indexes
    if _chunks is None:
        with CHUNKS_PATH.open() as f:
            _chunks = [json.loads(line) for line in f]
        _indexes = build_indexes(_chunks)
    return _chunks, _indexes


def _dispatch(name: str, tool_input: dict) -> dict:
    """Route one model-requested tool call to its Python implementation.

    Each branch pulls only the arguments that tool needs out of `tool_input`
    (the tools take different shapes, so a generic `**tool_input` unpack
    isn't safe -- e.g. `retrieve_policy` also needs the shared index, which
    isn't part of the model's input at all). Missing required keys raise
    `KeyError`/`TypeError` here, which `run_claim()` catches around this
    call and turns into a tool-result error the model can see and react to.
    """
    if name == "extract_fields":
        return run_extract_fields(tool_input["claim_text"])
    if name == "retrieve_policy":
        chunks, indexes = _get_index()
        return run_retrieve_policy(
            tool_input["query"], chunks, indexes,
            insurer=tool_input.get("insurer"), product=tool_input.get("product"),
        )
    if name == "get_policy_certificate":
        return run_get_policy_certificate(tool_input["insurer"])
    if name == "calculate_payout":
        return run_calculate_payout(
            tool_input["claim_amount"], tool_input["excess"], tool_input["coverage_limit"],
        )
    if name == "escalate_to_human":
        return run_escalate_to_human(tool_input["reason"])
    raise ValueError(f"Unknown tool: {name}")


def _parse_decision(text: str) -> str:
    """Pull the final approve/deny/escalate label out of the model's reply.

    The system prompt requires the reply to START WITH 'DECISION: <label>',
    so this checks a prefix, not just substring containment -- a plain `in`
    check would misfire on reasoning text that merely mentions another label
    (e.g. "I won't say DECISION: approve here, escalating instead" contains
    the substring "DECISION: approve" but is not that decision). Any reply
    that doesn't start with a recognized label falls back to the safe
    default, 'escalate', rather than guessing.
    """
    stripped = text.strip()
    for label in ("approve", "deny", "escalate"):
        if stripped.startswith(f"DECISION: {label}"):
            return label
    return "escalate"


def run_claim(claim_text: str) -> dict:
    """Triage one claim narrative to a final approve/deny/escalate decision.

    Drives the tool-use conversation turn by turn: send the running message
    history, execute whatever tools the model asked for, feed the results
    back, repeat until the model replies with plain text (its decision) or
    `MAX_TURNS` is hit. Every turn -- tool call or final text -- is recorded
    in `turns` so the full reasoning trace can be audited later from the
    decision log, not just the final answer.

    Returns a dict with `turns` (full trace), `final_decision`
    (approve/deny/escalate), `reasoning` (the model's stated reasoning, or
    the escalation reason), and `citations` (accumulated from every
    `retrieve_policy` call, regardless of which turn made them).
    """
    messages = [{"role": "user", "content": f"Triage this claim:\n\n{claim_text}"}]
    turns = []
    citations: list[dict] = []

    for turn_count in range(1, MAX_TURNS + 1):
        response = _client().messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # No more tool calls -- the model has committed to a final decision.
        if response.stop_reason != "tool_use":
            text = next((b.text for b in response.content if b.type == "text"), "")
            turns.append({"turn": turn_count, "type": "final", "text": text})
            return {
                "turns": turns,
                "final_decision": _parse_decision(text),
                "reasoning": text,
                "citations": citations,
            }

        # Otherwise execute every tool call this turn requested and report
        # results back in the same order, keyed by tool_use_id (required by
        # the API to match results to their calls when a turn has several).
        tool_results = []
        escalate_reason = None
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = _dispatch(block.name, block.input)
            except Exception as e:
                # A malformed tool call (e.g. a required arg the model
                # forgot) becomes a visible error the model can react to,
                # instead of crashing the whole claim.
                result = {"error": str(e)}

            turns.append({"turn": turn_count, "type": "tool_call", "tool": block.name, "input": block.input, "output": result})

            if block.name == "retrieve_policy" and "citations" in result:
                citations.extend(result["citations"])
            if block.name == "escalate_to_human":
                # Captured here, at the source, rather than re-scanning
                # `turns` after the loop -- also avoids a KeyError if the
                # dispatch above failed and `result` has no "reason" key.
                escalate_reason = result.get("reason") or result.get("error") or "escalated (no reason given)"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        if escalate_reason is not None:
            return {
                "turns": turns,
                "final_decision": "escalate",
                "reasoning": escalate_reason,
                "citations": citations,
            }

    # Hit MAX_TURNS without a decision or an explicit escalation -- force
    # one rather than returning an unfinished/ambiguous state.
    turns.append({"turn": MAX_TURNS, "type": "forced_escalation", "reason": "max turns exceeded"})
    return {
        "turns": turns,
        "final_decision": "escalate",
        "reasoning": "max turns exceeded",
        "citations": citations,
    }


if __name__ == "__main__":
    # Smoke test: _parse_decision must key off a *prefix* match, not a
    # substring match -- this exact case ("DECISION: approve" appears in the
    # text but not as the reply's opening) is what the old substring-based
    # version got wrong.
    cases = [
        ("DECISION: approve\n\nCovered under the storm peril, no exclusions found.", "approve"),
        ("DECISION: deny -- cause matches a named exclusion.", "deny"),
        ("DECISION: escalate -- policy text is ambiguous on this peril.", "escalate"),
        ("I won't say DECISION: approve here -- escalating instead.", "escalate"),
        ("no decision keyword at all", "escalate"),
    ]
    for text, expected in cases:
        got = _parse_decision(text)
        assert got == expected, f"_parse_decision({text!r}) = {got!r}, expected {expected!r}"
    print(f"{len(cases)}/{len(cases)} _parse_decision smoke tests passed")
