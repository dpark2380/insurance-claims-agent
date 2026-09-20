"""Phase 4, Step 4.3 -- fairness guardrails. SCAFFOLD ONLY, not implemented.

Goal: 5-8 paired claims, identical facts, differing only in a
protected-attribute-adjacent detail (claimant age mentioned or not, suburb
swapped between a high- and low-income-coded area). Any pair whose
final_decision differs is a finding.
"""

from __future__ import annotations

# TODO: each pair should be identical except for one protected-attribute-
# adjacent detail -- keep everything else (cause, amount, dates) byte-for-
# byte the same so any decision difference is attributable to that one swap.
FAIRNESS_PAIRS: list[dict] = [
    # {"pair_id": "age-001", "variant_a": "...", "variant_b": "...", "attribute_varied": "age mentioned"},
]

# TODO: keyword/phrase list for the post-hoc reasoning scan -- protected-
# attribute language that shouldn't appear as a stated factor in reasoning
# text even if it doesn't change the decision.
PROTECTED_ATTRIBUTE_TERMS: list[str] = []


def run_pair(pair: dict) -> dict:
    """TODO: run both variants through agent.loop.run_claim, compare
    final_decision, flag if they differ."""
    raise NotImplementedError


def scan_reasoning_for_flags(reasoning_text: str) -> list[str]:
    """TODO: simple keyword scan over PROTECTED_ATTRIBUTE_TERMS -- flag
    (don't block) any hit for human audit."""
    raise NotImplementedError


def main():
    """TODO: run every pair, write differing-decision findings to
    outputs/fairness_findings.md."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
