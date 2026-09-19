# Writing `code/extract/extract_fields.py` — detailed guide

Goal: one function, `extract_fields(claim_text: str) -> ClaimExtraction`, that Phase 3's
agent will call once per claim. Internally: try the local LoRA r=4 adapter first (free,
already matches zero-shot on 4/6 fields per `phase2_findings.md`), fall back to a live
Claude call if the adapter's output doesn't parse/validate, and raise if somehow neither
works.

---

## 1. Imports and constants

You'll need:
- `json` (stdlib)
- `torch`
- from `peft`: `PeftModel`
- from `transformers`: `AutoModelForCausalLM`, `AutoTokenizer`
- from `extract.schema`: `ClaimExtraction`
- from `extract.zero_shot`: `SYSTEM_PROMPT`, `_strip_fences`, and either `extract_one` directly, or just `_client`/`MODEL` from `rag.generate` if you'd rather call the API yourself here
- The winning adapter path: `Path(__file__).resolve().parents[2] / "outputs" / "lora-claims-extractor-r4"` — same `ROOT`-relative pattern every other file in this project uses.

## 2. Cache the model instead of loading it per call

This is the part most likely to be skipped if you're moving fast, and it'll make the
whole pipeline unusably slow if you do. The project already has the exact pattern to
copy for this — look at `_client()` in `rag/generate.py`: a module-level variable
starting as `None`, and a small function that builds the real object only the first
time it's called, then returns the cached one on every later call.

Do the same thing here, but caching a `(model, tokenizer)` pair instead of an API
client — build it once with `AutoTokenizer.from_pretrained` + `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16).to("mps")` + `PeftModel.from_pretrained(base, adapter_path)`, exactly like `lora_infer.py`'s `load_model` already does. Roughly:

```
_model = None
_tokenizer = None

def _get_model():
    global _model, _tokenizer
    if _model is None:
        # build once, exactly like lora_infer.load_model(4)
        ...
    return _model, _tokenizer
```

## 3. The LoRA attempt — reuse, don't rewrite

The actual generation logic (build the chat messages, `apply_chat_template` with
`add_generation_prompt=True`, remember to pull `["input_ids"]` out of the returned
`BatchEncoding`, `model.generate(...)`, decode, `_strip_fences`, `json.loads`,
`ClaimExtraction.model_validate`) is already written and working in `lora_infer.py`'s
`extract_one`. Don't reimplement it — either import and call that function directly
(it already returns `(dict | None, elapsed)`, which is exactly the "did it work"
signal you need), or factor the shared logic out if you want a cleaner split. Simplest
path: just import `extract_one` from `extract.lora_infer` and call it with the cached
model/tokenizer from step 2.

## 4. The Claude fallback — same idea, different existing function

`zero_shot.py`'s `extract_one` already does the equivalent thing against the Claude
API (same `SYSTEM_PROMPT`, same parse/validate/`None`-on-failure contract). If the
LoRA attempt returns `None`, call that one instead, passing the same `claim_text`.

## 5. Converting the winning dict into the promised return type

Whichever path succeeds hands you back a plain dict (`.model_dump(mode="json")`
already applied inside both existing `extract_one` functions). The function signature
promises a `ClaimExtraction` object, not a dict — so the very last step, regardless of
which path produced the result, is `ClaimExtraction.model_validate(winning_dict)`,
and *that's* what you return.

## 6. Handle the "both failed" case explicitly

Should be rare (0 parse failures across every LoRA r=4 eval run so far), but a
function whose signature promises a `ClaimExtraction` shouldn't silently return `None`
or an empty object if both attempts come back empty. Look at how `rag/generate.py`
handles the equivalent situation (no usable text block came back at all) — it raises
a `RuntimeError` with a clear message rather than returning something fake. Match that
shape here: if both the LoRA and the Claude fallback return `None`, raise rather than
inventing a placeholder result.

---

## Verification checklist

- [ ] Call `extract_fields()` twice in a row in a quick manual test and confirm the
      second call doesn't reload the model from disk (should be near-instant compared
      to the first call, which pays the one-time load cost).
- [ ] Manually force the LoRA path to fail (e.g. temporarily point it at a broken
      adapter path, or feed it something you know it'll mis-parse) and confirm the
      Claude fallback actually engages rather than the function just returning `None`
      or crashing.
- [ ] Confirm the return type is genuinely `ClaimExtraction` — e.g. `isinstance(result, ClaimExtraction)` — not a dict that happens to look right.
- [ ] Run it over a handful of real rows from `test.jsonl` and sanity-check the output
      against `gold_extraction` by eye, the same kind of check already done throughout
      this project rather than trusting "it ran without an error."

## Common pitfalls

- Reloading the model inside the function body instead of caching it at module level
  — works, but makes every single call to `extract_fields()` pay the multi-second
  model-load cost, defeating the whole point of the local-model path being cheap.
- Returning the raw dict from whichever `extract_one` succeeded instead of
  re-validating it into a `ClaimExtraction` — will satisfy casual testing but breaks
  the actual promised interface Phase 3 is built to depend on.
- Forgetting the `["input_ids"]` extraction from `apply_chat_template`'s return value
  if you end up writing the generation logic yourself instead of importing
  `lora_infer.extract_one` — this project has already hit that exact bug twice.
