import json
import torch

from peft import PeftModel
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from extract.schema import ClaimExtraction
from extract.zero_shot import SYSTEM_PROMPT, _strip_fences, extract_one
from extract.lora_infer import extract_one as lora_extract_one

# r=4, r=8, and r=16 are statistically indistinguishable on held-out accuracy
# (see outputs/phase2_findings.md) -- r=4 is used because it's the cheapest
# of three tied ranks, not because it scored best.
ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "outputs" / "lora-claims-extractor-r4"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_model = None
_tokenizer = None

def _get_model():
    global _model, _tokenizer
    if _model is None:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("mps")
        _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
        _model.eval()

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

    return _model, _tokenizer

def extract_fields(claim_text):
    model, tokenizer = _get_model()

    # Trying LoRA Adapter first.
    lora_result, _elapsed = lora_extract_one(model, tokenizer, claim_text)

    if lora_result is not None:
        winning_dict = lora_result
    else:
        # Fall back to Claude.
        claude_result, _usage = extract_one(claim_text)
        if claude_result is not None:
            winning_dict = claude_result
        else:
            raise RuntimeError(f"extract_fields: both LoRA and Claude failed to produce a valid extraction for: {claim_text!r}")

    return ClaimExtraction.model_validate(winning_dict)

