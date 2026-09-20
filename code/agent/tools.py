"""Tool schemas + implementations for the Phase 3 claims-triage agent.

Four tools, each a thin wrapper over an existing Phase 1/2 function -- no new
extraction or retrieval logic lives here, only the glue an LLM tool-use loop
needs (JSON-serializable in/out, hand-written schemas the model reads to
decide when to call each one).
"""

from __future__ import annotations

import json
from pathlib import Path

from extract.extract_fields import extract_fields
from rag.generate import generate_answer
from rag.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES_PATH = ROOT / "data" / "claims" / "policy_certificates.json"

TOOLS = [
    {
        "name": "extract_fields",
        "description": (
            "Extract structured fields (claim_type, date_of_loss, cause, "
            "damaged_item, estimated_amount, policy_number) from a raw claim "
            "narrative. Call this first on every claim -- the other tools "
            "need these fields to do their job."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_text": {"type": "string", "description": "The raw claim narrative."},
            },
            "required": ["claim_text"],
        },
    },
    {
        "name": "retrieve_policy",
        "description": (
            "Answer a coverage, exclusion, or peril-definition question against "
            "real policy (PDS/KFS) text, with citations. Use this to check "
            "whether a claim's cause is covered or an exclusion applies. This "
            "does NOT return excess or coverage-limit figures -- those are "
            "policyholder-specific and only available from "
            "get_policy_certificate. Only pass insurer/product if the claim "
            "narrative names one "
            "explicitly -- otherwise leave them out and it searches every "
            "insurer's policy text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The coverage/exclusion/excess question to ask."},
                "insurer": {"type": "string", "description": "Optional insurer name filter, e.g. 'AAMI'."},
                "product": {"type": "string", "description": "Optional product filter, e.g. 'Home', 'Contents'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_policy_certificate",
        "description": (
            "Look up a policyholder's certificate-of-insurance figures "
            "(excess, coverage_limit) for a named insurer. These are "
            "customer-specific numbers that never appear in PDS/KFS text, so "
            "this is the only source for the excess/coverage_limit inputs "
            "calculate_payout needs. Requires a specific insurer -- if the "
            "claim doesn't name one, you cannot look this up and must not "
            "guess a number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "insurer": {"type": "string", "description": "The insurer named in the claim, e.g. 'AAMI'."},
            },
            "required": ["insurer"],
        },
    },
    {
        "name": "calculate_payout",
        "description": (
            "Compute the payout amount given a claimed amount, the policy's "
            "excess, and its coverage limit (sum insured). Pure arithmetic -- "
            "call get_policy_certificate first to find the real excess and "
            "coverage limit numbers, never guess them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "claim_amount": {"type": "number"},
                "excess": {"type": "number"},
                "coverage_limit": {"type": "number"},
            },
            "required": ["claim_amount", "excess", "coverage_limit"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Stop and hand this claim to a human reviewer instead of "
            "deciding. You MUST call this -- not guess -- whenever "
            "retrieve_policy returns contradictory or ambiguous coverage "
            "text, extract_fields returns a null estimated_amount that "
            "cannot be resolved any other way, or the claimed cause directly "
            "conflicts with a policy exclusion you found. This ends the "
            "claim's processing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Specific, concrete reason for escalating."},
            },
            "required": ["reason"],
        },
    },
]


def run_extract_fields(claim_text: str) -> dict:
    """Thin JSON-serializable wrapper over Phase 2's `extract_fields()`.

    Converts the Pydantic `ClaimExtraction` model to a plain dict (`mode`
    "json" so `date`/`Enum` fields become strings, not Python objects) since
    tool results are serialized straight to JSON for the API.
    """
    return extract_fields(claim_text).model_dump(mode="json")


def run_retrieve_policy(query: str, chunks: list[dict], indexes: dict, insurer: str | None = None, product: str | None = None) -> dict:
    """Answer a coverage question against real policy text, with citations.

    `chunks`/`indexes` are passed in (built once by the caller, see
    `loop._get_index`) rather than rebuilt here on every call.
    """
    retrieved = retrieve(query, chunks, indexes, insurer=insurer, product=product)
    answer = generate_answer(query, retrieved)
    # One citation per retrieved chunk, not per fact the answer actually
    # used -- callers see everything that was in context, not just what got
    # cited in the generated text.
    citations = [
        {"insurer": c["insurer"], "product": c["product"], "doc_type": c["doc_type"], "page": c["page"]}
        for c in retrieved
    ]
    return {"answer": answer, "citations": citations}


_certificates = None


def run_get_policy_certificate(insurer: str) -> dict:
    """Look up a policyholder's excess/coverage_limit for a named insurer.

    `_certificates` is loaded from disk once per process and cached, same
    pattern as `loop._get_index` -- a handful of insurers never changes
    mid-run. Returns a *copy* of the cached record (`dict(cert)`) so a
    caller mutating the result can't corrupt the shared cache for later
    calls.
    """
    global _certificates
    if _certificates is None:
        _certificates = json.loads(CERTIFICATES_PATH.read_text())
    cert = _certificates.get(insurer)
    if cert is None:
        return {"error": f"No certificate on file for insurer {insurer!r}"}
    return dict(cert)


def run_calculate_payout(claim_amount: float, excess: float, coverage_limit: float) -> dict:
    """Pure arithmetic, no LLM: payout = claimed amount, capped at the
    coverage limit, minus the excess -- never negative."""
    payout = max(0.0, min(claim_amount, coverage_limit) - excess)
    return {"payout": payout}


def run_escalate_to_human(reason: str) -> dict:
    """Terminal marker the agent loop checks for -- no side effects beyond
    the returned record, which the loop uses as the claim's final reasoning."""
    return {"status": "escalated", "reason": reason}


if __name__ == "__main__":
    # Smoke test: pure-arithmetic payout cases, including the two boundary
    # rules ("never negative" and "capped at coverage_limit") that are easy
    # to get backwards when adjusting this formula later.
    cases = [
        # (claim_amount, excess, coverage_limit) -> expected payout
        (2500.0, 500.0, 50000.0, 2000.0),   # ordinary case: claim - excess
        (100.0, 500.0, 50000.0, 0.0),       # claim smaller than excess -> never negative
        (90000.0, 500.0, 50000.0, 49500.0),  # claim above coverage_limit -> capped first
    ]
    for claim_amount, excess, coverage_limit, expected in cases:
        got = run_calculate_payout(claim_amount, excess, coverage_limit)["payout"]
        assert got == expected, f"run_calculate_payout({claim_amount}, {excess}, {coverage_limit}) = {got}, expected {expected}"
    print(f"{len(cases)}/{len(cases)} run_calculate_payout smoke tests passed")
