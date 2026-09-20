"""Synthetic claim narrative + gold-label generation for Phase 2 extraction (step 2.2).

Run (from code/):
  python -m extract.generate_synthetic generate   # append to data/claims/raw.jsonl (resumable)
  python -m extract.generate_synthetic spotcheck  # sample 20 -> outputs/synthetic_spotcheck.jsonl
  python -m extract.generate_synthetic split       # 70/15/15 -> data/claims/{train,val,test}.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from extract.schema import ClaimExtraction, ClaimType  # noqa: E402
from rag.generate import MODEL, _client  # noqa: E402

RAW_PATH = ROOT / "data" / "claims" / "raw.jsonl"
SPOTCHECK_PATH = ROOT / "outputs" / "synthetic_spotcheck.jsonl"
TRAIN_PATH = ROOT / "data" / "claims" / "train.jsonl"
VAL_PATH = ROOT / "data" / "claims" / "val.jsonl"
TEST_PATH = ROOT / "data" / "claims" / "test.jsonl"

TARGET = 300
CLAIM_TYPES = list(ClaimType)
COMPLEXITY_HINTS = [
    "a clean, single-item claim with no ambiguity",
    "a claim where the cause is somewhat ambiguous or hard to pin down",
    "a claim involving multiple damaged items",
    "a claim where the claimant doesn't mention a policy number",
    "a claim where the claimant hasn't got a repair estimate yet, so no dollar amount",
]
# Independent axis from COMPLEXITY_HINTS -- varies HOW the claimant writes, not WHAT
# happened. Without this every narrative was written in one uniform "clean Claude
# prose" register, so a model trained on it risks learning that register rather than
# real claimant phrasing variance. len()=4 vs COMPLEXITY_HINTS' 5 means the two axes
# fall out of phase every iteration and cover all 20 combinations every 20 rows.
REGISTER_HINTS = [
    "clean, professional writing",
    "informal and casual, with a couple of minor typos or dropped punctuation",
    "terse and a bit incomplete, as if dashed off quickly, but every field the schema "
    "needs must still be genuinely stated or clearly implied -- terse is not an excuse "
    "to be ambiguous",
    "written the way a non-native English speaker might phrase it -- simpler sentence "
    "structure, occasional slightly unusual word choice -- but still perfectly clear "
    "and grammatical enough to read",
]

SCHEMA_JSON = json.dumps(ClaimExtraction.model_json_schema(), indent=2)

SYSTEM_PROMPT = f"""You write short, realistic synthetic training examples for an
insurance claim field-extraction system covering Australian home insurance claims.

For each request, invent ONE realistic AU home-insurance claim scenario and return
a single JSON object with exactly two keys:

"narrative": a first-person claimant narrative (2-5 sentences, plain conversational
English) describing what happened, as if reporting the loss to their insurer for the
first time. Do not mention insurers, disputes, investigations, or outcomes -- this is
a first report of a loss, before any of that has happened.

"gold_extraction": the structured fields for that exact narrative, matching this JSON
schema exactly:
{SCHEMA_JSON}

Rules:
- Every fact in gold_extraction must be stated or clearly implied in narrative --
  never add a field value that isn't grounded in the narrative text.
- The narrative MUST state the full date of loss explicitly -- day, month, and year
  (e.g. "on 14 March 2024" or "on March 14, 2024"). Never write a vague date ("last
  month", "the other week", "on the 14th") -- date_of_loss in gold_extraction must be
  readable directly off the narrative's own words, not invented to fill the field.
- estimated_amount and policy_number may be null if the narrative doesn't mention
  them (real claimants often omit these).
- date_of_loss must be a real ISO date (YYYY-MM-DD), within the last 3 years.
- claim_type "flood" applies ONLY when water escaped the normal confines of a lake,
  river, creek, dam, or other named natural watercourse -- if you generate a flood
  example, the narrative must name that watercourse. Any other rain-caused scenario
  (wind-driven rain, roof ingress, drainage overwhelm, surface pooling with no named
  watercourse) must be labelled "storm" instead, regardless of rainfall volume.
- Write the narrative in the requested register (see the "register" hint in the
  request), varying vocabulary and sentence structure accordingly -- but the
  groundedness rule above still applies without exception: every gold_extraction
  field must still be genuinely readable off the narrative. A messier register is a
  style change, never an excuse to make the underlying facts ambiguous or missing.
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


def generate_one(claim_type: ClaimType, complexity: str, register: str = REGISTER_HINTS[0]) -> dict | None:
    user_msg = (
        f"Generate one example. claim_type: {claim_type.value}. "
        f"complexity: {complexity}. register: {register}."
    )
    last_error: Exception | None = None
    for _attempt in range(2):
        response = _client().messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            output_config={"effort": "low"},
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            last_error = RuntimeError(f"no text block (stop_reason={response.stop_reason!r})")
            continue
        try:
            parsed = json.loads(_strip_fences(text))
            gold = ClaimExtraction.model_validate(parsed["gold_extraction"])
            narrative = parsed["narrative"]
            if not isinstance(narrative, str) or not narrative.strip():
                raise ValueError("empty or missing narrative")
            return {"narrative": narrative, "gold_extraction": gold.model_dump(mode="json")}
        except Exception as e:  # noqa: BLE001 -- deliberately broad: any parse/validation failure retries once
            last_error = e
    print(f"  skip (claim_type={claim_type.value}): {last_error}", file=sys.stderr)
    return None


def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in RAW_PATH.open()) if RAW_PATH.exists() else 0
    if existing >= TARGET:
        print(f"Already have {existing}/{TARGET} examples in {RAW_PATH} -- nothing to do.")
        return

    accepted, skipped, i = existing, 0, existing
    with RAW_PATH.open("a") as f:
        while accepted < TARGET:
            claim_type = CLAIM_TYPES[i % len(CLAIM_TYPES)]
            complexity = COMPLEXITY_HINTS[i % len(COMPLEXITY_HINTS)]
            register = REGISTER_HINTS[i % len(REGISTER_HINTS)]
            i += 1
            row = generate_one(claim_type, complexity, register)
            if row is None:
                skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            accepted += 1
            print(f"[{accepted}/{TARGET}] {claim_type.value}")

    print(f"\nDone. {accepted} accepted, {skipped} skipped/failed. -> {RAW_PATH}")


def write_spotcheck():
    lines = RAW_PATH.read_text().splitlines()
    random.seed(42)
    sample = random.sample(lines, min(20, len(lines)))
    SPOTCHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPOTCHECK_PATH.write_text("\n".join(sample) + "\n")
    print(f"Wrote {len(sample)} rows to {SPOTCHECK_PATH} -- read these by hand before splitting.")


def write_split():
    # Stratified per claim_type -- a plain shuffle-then-slice can leave a class with
    # 0-1 rows in test purely by chance, even when the source pool is balanced.
    lines = RAW_PATH.read_text().splitlines()
    by_class: dict[str, list[str]] = {}
    for line in lines:
        ct = json.loads(line)["gold_extraction"]["claim_type"]
        by_class.setdefault(ct, []).append(line)

    random.seed(42)
    train, val, test = [], [], []
    for _ct, rows in sorted(by_class.items()):
        shuffled = rows[:]
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * 0.7)
        n_val = int(len(shuffled) * 0.15)
        train += shuffled[:n_train]
        val += shuffled[n_train:n_train + n_val]
        test += shuffled[n_train + n_val:]

    for rows in (train, val, test):
        random.shuffle(rows)

    for path, rows in [(TRAIN_PATH, train), (VAL_PATH, val), (TEST_PATH, test)]:
        path.write_text("\n".join(rows) + "\n")

    print(f"train={len(train)} val={len(val)} test={len(test)} (total {len(lines)})")
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(json.loads(r)["gold_extraction"]["claim_type"] for r in rows)
        print(f"  {name}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "spotcheck", "split"], nargs="?", default="generate")
    args = parser.parse_args()
    {"generate": main, "spotcheck": write_spotcheck, "split": write_split}[args.command]()
