"""LoRA-adapter inference for claim field extraction (step 2.5).

Run (from code/):
  python -m extract.lora_infer --rank 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEST_PATH = ROOT / "data" / "claims" / "test.jsonl"
AFCA_TEST_PATH = ROOT / "data" / "claims" / "afca_test.jsonl"
OUTPUTS_DIR = ROOT / "outputs"

from extract.eval import score  # noqa: E402
from extract.schema import ClaimExtraction  # noqa: E402
from extract.zero_shot import SYSTEM_PROMPT, _strip_fences  # noqa: E402 -- same instruction as the zero-shot baseline


def load_model(rank: int):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("mps")
    adapter_dir = OUTPUTS_DIR / f"lora-claims-extractor-r{rank}"
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


def extract_one(model, tokenizer, narrative: str) -> tuple[dict | None, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": narrative},
    ]
    # apply_chat_template returns a BatchEncoding (dict-like) on this transformers
    # version even with return_tensors="pt" -- pull the actual tensor out of it.
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )["input_ids"].to("mps")

    start = time.monotonic()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=300, do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.monotonic() - start

    completion_ids = output_ids[0][input_ids.shape[-1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    try:
        parsed = json.loads(_strip_fences(text))
        validated = ClaimExtraction.model_validate(parsed)
        return validated.model_dump(mode="json"), elapsed
    except Exception:  # noqa: BLE001 -- any parse/validation failure counts as wrong, not skipped
        return None, elapsed


def main(rank: int, dataset: str):
    data_path = AFCA_TEST_PATH if dataset == "afca" else TEST_PATH
    suffix = "_afca" if dataset == "afca" else ""

    model, tokenizer = load_model(rank)
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    gold = [r["gold_extraction"] for r in rows]

    predictions, latencies = [], []
    for i, row in enumerate(rows, 1):
        pred, elapsed = extract_one(model, tokenizer, row["narrative"])
        predictions.append(pred)
        latencies.append(elapsed)
        print(f"[{i}/{len(rows)}] {'ok' if pred is not None else 'PARSE FAIL'} ({elapsed:.2f}s)")

    metrics = score(predictions, gold)
    avg_latency = sum(latencies) / len(latencies)

    out_path = OUTPUTS_DIR / f"lora_r{rank}{suffix}_results.json"
    out_path.write_text(json.dumps(
        {"rank": rank, "dataset": dataset, **metrics, "avg_latency_s": avg_latency,
         "predictions": predictions, "gold": gold},
        indent=2,
    ))

    print(f"\noverall={metrics['overall_mean_of_fields']:.3f} parse_failures={metrics['parse_failures']}/{metrics['n']} "
          f"avg_latency={avg_latency:.2f}s")
    print(f"per_field={ {k: round(v, 3) for k, v in metrics['per_field'].items()} }")
    print(f"-> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--dataset", choices=["test", "afca"], default="test")
    args = parser.parse_args()
    main(args.rank, args.dataset)
