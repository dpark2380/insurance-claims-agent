"""Per-field scoring for claim extraction predictions against gold labels (step 2.3).

None of ClaimExtraction's fields are actually list-typed, so "F1 for anything
list-like" is applied here as token-set F1 on the two free-text fields (cause,
damaged_item) -- soft partial credit instead of brittle exact-string matching,
same spirit as list-F1 without a literal list to compare.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokenize(pred), _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = len(pred_tokens & gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _amount_match(pred: float | None, gold: float | None, tolerance: float = 0.05) -> float:
    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    if gold == 0:
        return 1.0 if pred == 0 else 0.0
    return 1.0 if abs(pred - gold) / abs(gold) <= tolerance else 0.0


def _exact_match(pred, gold) -> float:
    return 1.0 if pred == gold else 0.0


FIELD_SCORERS = {
    "claim_type": _exact_match,
    "date_of_loss": _exact_match,
    "cause": _token_f1,
    "damaged_item": _token_f1,
    "estimated_amount": _amount_match,
    "policy_number": _exact_match,
}


def score_one(pred: dict | None, gold: dict) -> dict[str, float]:
    """Score one prediction against gold. pred=None (parse/validation failure) scores 0 on every field."""
    if pred is None:
        return {field: 0.0 for field in FIELD_SCORERS}
    return {field: scorer(pred.get(field), gold[field]) for field, scorer in FIELD_SCORERS.items()}


def score(predictions: list[dict | None], gold: list[dict]) -> dict:
    """Aggregate per-field scores across a whole test set."""
    if len(predictions) != len(gold):
        raise ValueError(f"predictions ({len(predictions)}) and gold ({len(gold)}) length mismatch")

    per_record = [score_one(p, g) for p, g in zip(predictions, gold)]
    fields = list(FIELD_SCORERS)
    per_field = {f: sum(r[f] for r in per_record) / len(per_record) for f in fields}
    return {
        "per_field": per_field,
        "overall_mean_of_fields": sum(per_field.values()) / len(per_field),
        "n": len(predictions),
        "parse_failures": sum(1 for p in predictions if p is None),
    }


if __name__ == "__main__":
    gold = [{"claim_type": "storm", "date_of_loss": "2022-03-01", "cause": "wind damaged the roof",
             "damaged_item": "roof tiles", "estimated_amount": 5000.0, "policy_number": "HM-123"}]
    cases = {
        "exact": [dict(gold[0])],
        "close": [{"claim_type": "storm", "date_of_loss": "2022-03-01", "cause": "wind damage to roof",
                    "damaged_item": "roof", "estimated_amount": 5100.0, "policy_number": "HM-123"}],
        "wrong": [{"claim_type": "fire", "date_of_loss": "2020-01-01", "cause": "unrelated",
                    "damaged_item": "unrelated", "estimated_amount": 1.0, "policy_number": None}],
        "parse_fail": [None],
    }
    expected_overall = {"exact": 1.0, "wrong": 0.0, "parse_fail": 0.0}  # "close" checked by inspection below

    for name, preds in cases.items():
        r = score(preds, gold)
        print(f"{name}: overall={r['overall_mean_of_fields']:.3f} per_field={ {k: round(v, 2) for k, v in r['per_field'].items()} }")
        if name in expected_overall:
            assert r["overall_mean_of_fields"] == expected_overall[name], f"{name} smoke test failed"
    print("\nsmoke tests passed")
