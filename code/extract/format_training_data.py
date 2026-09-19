"""Convert train/val splits into instruction-formatted pairs for LoRA fine-tuning (step 2.4).

Run (from code/):
  python -m extract.format_training_data
"""

from __future__ import annotations

import json
from pathlib import Path

from extract.zero_shot import SYSTEM_PROMPT as INSTRUCTION  # same wording the zero-shot baseline
# uses, so the fine-tune and the baseline are trained/scored against an identical task description.

ROOT = Path(__file__).resolve().parents[2]
SPLITS = {
    "train": (ROOT / "data" / "claims" / "train.jsonl", ROOT / "data" / "claims" / "train_formatted.jsonl"),
    "val": (ROOT / "data" / "claims" / "val.jsonl", ROOT / "data" / "claims" / "val_formatted.jsonl"),
}


def format_row(row: dict) -> dict:
    return {
        "instruction": INSTRUCTION,
        "input": row["narrative"],
        "output": json.dumps(row["gold_extraction"]),
    }


def main():
    for name, (src, dst) in SPLITS.items():
        rows = [json.loads(line) for line in src.read_text().splitlines()]
        formatted = [format_row(r) for r in rows]
        dst.write_text("\n".join(json.dumps(f) for f in formatted) + "\n")
        print(f"{name}: {len(formatted)} rows -> {dst}")


if __name__ == "__main__":
    main()
