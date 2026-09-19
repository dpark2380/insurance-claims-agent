"""Zero-shot Claude baseline for claim field extraction (step 2.3).

Run (from code/):
  python -m extract.zero_shot --dataset test   # data/claims/test.jsonl (synthetic)
  python -m extract.zero_shot --dataset afca   # data/claims/afca_test.jsonl (real-world)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from extract.eval import score  # noqa: E402
from extract.schema import ClaimExtraction  # noqa: E402
from rag.generate import MODEL, _client  # noqa: E402

TEST_PATH = ROOT / "data" / "claims" / "test.jsonl"
AFCA_TEST_PATH = ROOT / "data" / "claims" / "afca_test.jsonl"
RESULTS_PATH = ROOT / "outputs" / "zero_shot_results.json"
AFCA_RESULTS_PATH = ROOT / "outputs" / "zero_shot_results_afca.json"

# Sonnet 5 pricing, $/token (see the claude-api skill's pricing table for the source figures).
INPUT_PRICE = 2.00 / 1_000_000
OUTPUT_PRICE = 10.00 / 1_000_000

SCHEMA_JSON = json.dumps(ClaimExtraction.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""Extract structured fields from an Australian home-insurance claim
narrative. Return a single JSON object matching this schema exactly:
{SCHEMA_JSON}

Rules:
- Only extract what the narrative actually states or clearly implies -- never guess
  or invent a value.
- claim_type "flood" applies ONLY when water escaped the normal confines of a lake,
  river, creek, dam, or other named natural watercourse. Any other rain-caused loss
  (wind-driven rain, roof ingress, drainage overwhelm, surface water pooling with no
  named watercourse) is "storm", regardless of rainfall volume.
- estimated_amount and policy_number must be null if the narrative doesn't mention them.
- Output ONLY the JSON object, no other text, no markdown code fences."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.strip("`")
    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        text = rest if first_line.strip().lower() in ("", "json") else text
    return text.strip()


def extract_one(narrative: str) -> tuple[dict | None, dict]:
    """Returns (gold-shaped prediction dict, or None on parse/validation failure; usage/timing info)."""
    start = time.monotonic()
    response = _client().messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": narrative}],
        output_config={"effort": "low"},
    )
    elapsed = time.monotonic() - start
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_s": elapsed,
    }

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return None, usage
    try:
        parsed = json.loads(_strip_fences(text))
        validated = ClaimExtraction.model_validate(parsed)
        return validated.model_dump(mode="json"), usage
    except Exception:  # noqa: BLE001 -- any parse/validation failure counts as wrong, not skipped
        return None, usage


def _percentile(sorted_values: list[float], p: float) -> float:
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * p))
    return sorted_values[idx]


def main(dataset: str):
    data_path, results_path = (AFCA_TEST_PATH, AFCA_RESULTS_PATH) if dataset == "afca" else (TEST_PATH, RESULTS_PATH)

    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    gold = [r["gold_extraction"] for r in rows]

    predictions, usages = [], []
    for i, row in enumerate(rows, 1):
        pred, usage = extract_one(row["narrative"])
        predictions.append(pred)
        usages.append(usage)
        print(f"[{i}/{len(rows)}] {'ok' if pred is not None else 'PARSE FAIL'} ({usage['latency_s']:.2f}s)")

    metrics = score(predictions, gold)
    latencies = sorted(u["latency_s"] for u in usages)
    total_cost = sum(u["input_tokens"] * INPUT_PRICE + u["output_tokens"] * OUTPUT_PRICE for u in usages)

    result = {
        "dataset": dataset,
        "model": MODEL,
        **metrics,
        "cost_usd": total_cost,
        "latency_p50_s": _percentile(latencies, 0.5),
        "latency_p95_s": _percentile(latencies, 0.95),
        "predictions": predictions,
        "gold": gold,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2))

    print(f"\noverall={metrics['overall_mean_of_fields']:.3f} parse_failures={metrics['parse_failures']}/{metrics['n']} "
          f"cost=${total_cost:.4f} p50={result['latency_p50_s']:.2f}s p95={result['latency_p95_s']:.2f}s")
    print(f"per_field={ {k: round(v, 3) for k, v in metrics['per_field'].items()} }")
    print(f"-> {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["test", "afca"], default="test")
    args = parser.parse_args()
    main(args.dataset)
