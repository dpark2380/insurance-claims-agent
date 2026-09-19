# Writing `code/extract/train_lora.py` — detailed guide

Goal: LoRA fine-tune `Qwen/Qwen2.5-1.5B-Instruct` on `data/claims/train_formatted.jsonl`
(99 rows) to predict `gold_extraction` JSON from a claim narrative, validating against
`val_formatted.jsonl` (18 rows), on Apple Silicon (MPS).

`format_training_data.py` already produced the inputs — each line of both files has
an `instruction` (the extraction task description), an `input` (the narrative), and
an `output` (the gold JSON as a string).

---

## 1. Set up imports and paths

You'll need:
- `torch`
- from `peft`: `LoraConfig`, `get_peft_model`
- from `transformers`: `AutoModelForCausalLM`, `AutoTokenizer`, `Trainer`, `TrainingArguments`

Define constants for the base model name (`"Qwen/Qwen2.5-1.5B-Instruct"`), the two
formatted-data paths (`data/claims/train_formatted.jsonl`, `.../val_formatted.jsonl`),
and a max sequence length — pick a generous number for the latter (a short narrative
plus a small JSON object doesn't need much), or tokenize a handful of real rows first
and check their actual lengths if you want a measured number instead of a guess.

## 2. Build the dataset — this is the part most likely to go subtly wrong

The model must be graded **only** on producing the JSON output, not on "predicting"
its own instruction and narrative back. Skipping this doesn't cause an error — the
loss still goes down, training still finishes — it just trains on an easier, wrong
objective, which is exactly the kind of bug that survives past "it ran."

For each training row, you need two versions of the same conversation: the prompt
alone (system instruction + user narrative, rendered with whatever marks "now the
assistant is about to speak" but no assistant content yet), and the full conversation
(same, plus the assistant's JSON output). Tokenize both using the model's own chat
template, not a hand-built string — Qwen2.5-1.5B-Instruct expects its own specific
turn-formatting, and training against a different format fights its existing
instruction-following behavior instead of building on it.

The length of the prompt-alone tokenization tells you exactly where the completion
starts in the full tokenization. Build your training labels as a copy of the full
token sequence, then overwrite every position up through that boundary with the
"ignore this position" value the loss function recognizes, so gradient signal only
flows from the JSON tokens onward.

Before training anything, sanity-check this by decoding two things for one example:
the full token sequence, and only the tokens whose label wasn't overwritten. The first
should read as the whole conversation; the second should read as *only* the JSON
output. If narrative text shows up in the second one, the masking boundary is off by
something and needs fixing before you spend any training time on it.

Roughly, in pseudocode, for one row:

```
prompt_msgs = [system: instruction, user: input]
full_msgs   = prompt_msgs + [assistant: output]

prompt_ids  = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
full_ids    = tokenizer.apply_chat_template(full_msgs,   tokenize=True, add_generation_prompt=False)

labels = copy(full_ids)
labels[0 : len(prompt_ids)] = -100   # -100 = "ignore this position" for the loss

return {input_ids: full_ids, attention_mask: all-ones same length, labels: labels}
```

`add_generation_prompt=True` is what appends the "now the assistant is about to speak"
marker tokens without any assistant content yet — that's what makes `len(prompt_ids)`
line up exactly with where the completion starts in `full_ids`. This class should
subclass `torch.utils.data.Dataset`, reading the JSONL file once in `__init__` and
doing this tokenization per-row in `__getitem__`.

## 3. Load the model and tokenizer onto the GPU

Load the tokenizer, and if it doesn't already have a pad token defined, fall back to
using its end-of-sequence token as the pad token (`tokenizer.pad_token = tokenizer.eos_token`
when `tokenizer.pad_token is None`). Load the model in bfloat16 (keeps the ~3GB model's
memory footprint down) — `AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16)`
— then move it onto the Metal GPU directly with `.to("mps")`. Don't use the
`device_map=` argument for this; that's meant for splitting a model across several
devices or offloading to CPU, not for putting a whole model on one Apple GPU.

## 4. Wrap the model with the LoRA adapter

`LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")`,
then `model = get_peft_model(model, lora_config)`. Rank and alpha should be function
parameters, not hardcoded, so the same script can be re-run for the rank sweep later.

After wrapping, there's one easy-to-miss but important step: loop over `model.parameters()`
and, for every one where `param.requires_grad` is true (i.e. it belongs to the adapter,
not the frozen base), cast its `.data` to full precision (`.float()`) even though the
base model stays in bfloat16. This keeps the very small trainable matrices numerically
stable during optimization — skipping it doesn't error, it just risks quietly worse
training. Call `model.print_trainable_parameters()` afterward as a sanity check — it
should report a tiny fraction of the base model's 1.5B.

## 5. Configure the training run

Build a `TrainingArguments(...)` with, roughly:

```
per_device_train_batch_size = 1
per_device_eval_batch_size  = 1
gradient_accumulation_steps = 8
num_train_epochs            = <parameter, default 3>
eval_strategy                = "epoch"
logging_strategy            = "epoch"
save_strategy                = "no"
report_to                   = []
learning_rate                = 2e-4
output_dir                   = <somewhere under outputs/, doesn't matter much since save_strategy="no">
```

Batch size 1 sidesteps needing to write a custom padding strategy entirely, since no
two examples ever share a batch with matching lengths. Gradient accumulation gets you
a larger *effective* batch size without the memory cost of literally batching several
examples together. `eval_strategy="epoch"` + `logging_strategy="epoch"` (note: some
`transformers` versions call this argument `evaluation_strategy` instead — if you get
an unexpected-keyword error, that's a version difference, try the other name) is what
gets you the per-epoch train/val loss curve rather than just a final number.
`save_strategy="no"` skips intermediate checkpoints since you save the finished
adapter yourself. `report_to=[]` skips any wandb/experiment-tracker prompts.

Then `Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)`.

Don't turn on the trainer's own mixed-precision flags (`fp16=`/`bf16=` on
`TrainingArguments`) — the model is already loaded in bfloat16, so there's nothing
extra to gain, and mixed-precision support on Apple's GPU backend is inconsistent
across library versions.

## 6. Train, time it, and save

Wrap the actual training call (`result = trainer.train()`) with a wall-clock timer
(`time.monotonic()` before and after) — this measurement is what decides whether
running the same thing three more times (the rank sweep) is a five-minute detour or a
much bigger one. After training finishes, save both the adapter and the tokenizer —
`model.save_pretrained(out_dir)`, `tokenizer.save_pretrained(out_dir)` — to an output
directory whose name includes the rank you used (e.g. `outputs/lora-claims-extractor-r{rank}/`),
so a later sweep run doesn't silently overwrite this one. Print `result.training_loss`
and `trainer.state.log_history` — that history is what you actually read afterward to
answer the real question: is this rank and epoch count enough, too much, or about
right? Rising validation loss while training loss keeps dropping means it's starting
to memorize the 99 training examples; training loss still dropping steeply at the last
epoch means it probably hasn't finished learning yet.

## 7. Make it a runnable script

Accept rank, alpha, and epoch count as command-line arguments with the plan's defaults
(8, 16, 3). This is what turns the later rank sweep into three re-invocations of the
same script with a different rank argument, rather than new code each time.

---

## Verification checklist

- Decoded the masked-vs-unmasked view of one training example and confirmed the
  unmasked part is only the JSON, not the conversation.
- Printed the trainable-parameter count and confirmed it's a small fraction of 1.5B.
- After training, loaded the saved adapter fresh (in a separate script or session,
  not just reusing the in-memory object you just trained) — `peft.PeftModel.from_pretrained(base_model, out_dir)` —
  and ran one inference on a training example. If the output looks identical to what
  the un-adapted base model would produce, something's wrong — the adapter isn't
  actually changing behavior.
- Actually read the per-epoch train/val loss numbers before accepting the run as
  good, rather than treating "it finished without an error" as sufficient.

## Common pitfalls (none of these throw an error — they just silently do the wrong thing)

- Getting the masking boundary off by one, or mixing up which chat-template rendering
  option adds the "assistant about to speak" marker — training still runs, just on a
  slightly wrong target.
- Forgetting the pad-token fallback — usually harmless here since batch size is 1, but
  can surface unexpectedly depending on how the training library uses it internally.
- Reusing the same output directory across different ranks in the sweep — the second
  run's adapter silently overwrites the first's on disk.
