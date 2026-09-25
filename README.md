# Insurance Claims Agent

An agent that triages Australian home insurance claims: it extracts structured fields from a raw claim narrative, checks coverage against real insurer policy documents, calculates a payout with deterministic arithmetic over certificate-of-insurance figures, and decides to approve, deny, or escalate, citing the specific policy document and page behind each finding. Built to compare a fine-tuned local model against a zero-shot API baseline on cost, accuracy, and reliability, not just accuracy alone.

## Results

**Extraction (synthetic test set, n=54, per-field score blending exact match, 5% amount tolerance, and token-F1, see [Extraction metric](#extraction-metric))**

| | Zero-shot Claude | LoRA r=4 | LoRA r=8 | LoRA r=16 |
|---|---|---|---|---|
| Overall score | 0.930 | 0.822 | 0.822 | 0.823 |
| 95% CI (bootstrap) | [0.914, 0.945] | [0.767, 0.863] | [0.769, 0.863] | [0.770, 0.866] |
| Parse failures | 0/54 | 2/54 | 2/54 | 2/54 |

The three LoRA ranks are statistically indistinguishable from each other. Zero-shot's advantage over all three is real (its confidence interval does not overlap any of theirs). That rules out adapter capacity as the bottleneck: giving the local model more trainable parameters didn't close the gap, so the production choice comes down to cost, not accuracy (see [Why r=4](#why-r4-despite-a-lower-score)). The gap is uneven across fields, not one uniform drop: `date_of_loss` and `policy_number` are within 0.04 of zero-shot, `estimated_amount` within 0.07, but `claim_type` (0.13), `damaged_item` (0.13), and especially `cause` (0.25) lag further behind. The fields the payout calculation actually depends on (the date, the amount, the policy number used to look up the certificate) are the fields LoRA is closest to matching; the fields it's weakest on are the free-text description of what happened, which is exactly the failure mode the Claude fallback exists to catch.

**Agent cost and latency (20-claim batch run 4 times, 80 claim runs, full pipeline: extraction, retrieval, payout calculation, decision)**

| Run | Wall-clock latency/claim, mean (p95) | API latency/claim, mean | Cost/claim, mean |
|---|---|---|---|
| [outputs/](outputs/phase4_report.md) | 25.07s (45.29s) | 20.76s | $0.0342 |
| [run1](outputs/stability/run1/phase4_report.md) | 24.42s (40.00s) | 20.40s | $0.0392 |
| [run2](outputs/stability/run2/phase4_report.md) | 22.55s (41.38s) | 18.57s | $0.0348 |
| [run3](outputs/stability/run3/phase4_report.md) | 27.78s (81.02s) | 23.26s | $0.0366 |
| **Mean of 4 runs** | **24.96s** | **20.75s** | **$0.0362** |

Wall-clock latency is end to end per claim: local LoRA extraction, retrieval, tool execution, and every Claude call. API latency is the Claude calls alone. One-time startup (loading the LoRA model, building the index, about 9s) is reported separately, not charged to any claim. Cost includes every Claude call, including the extraction fallback, which fired once per run (claim-015, where both extractors fail). Measured with per-call token and latency instrumentation ([code/eval/instrumentation.py](code/eval/instrumentation.py)), not estimated. Pricing: `claude-sonnet-5` at $2/$10 per million input/output tokens.

This is the cost of the full agent, not a single model call: extraction, one or more `retrieve_policy` calls (each of which makes its own Claude call to generate a grounded answer), and the reasoning turns in between. At $0.036 per claim, 10,000 claims a month would cost roughly $360 (a simple multiplication of the measured mean, not a separately measured figure), most of it coming from the number of tool-calling turns a claim takes rather than from the underlying model's per-token price.

**Decision stability (same 20 claims, same code, 4 runs)**

12 of 20 claims got the same decision in all 4 runs; 8 flipped at least once. Escalations per run: 16, 12, 14, 16 of 20 (72.5% pooled). Decision accuracy against human labels has not been measured yet: blind gold labels go in [data/claims/batch20_gold.csv](data/claims/batch20_gold.csv), scored by [code/eval/decision_agreement.py](code/eval/decision_agreement.py).

**Retrieval quality (hit-rate@5 and MRR@5, n=8)**

| | Value |
|---|---|
| Hit-rate@5 | 1.000 (8/8) |
| MRR@5 | 0.613 |

Hit-rate@5 of 1.000 means the correct policy chunk was somewhere in the top 5 results for every one of the 8 questions. MRR@5 of 0.613 means it usually wasn't ranked first: an MRR of 1.0 would mean every correct chunk was the top result, and 0.613 corresponds to several questions where the right chunk placed 4th or 5th rather than 1st. In practice this means the generation step, which reads all 5 retrieved chunks rather than just the top one, is doing real work to find the right answer among plausible-looking distractors, not just passing through whatever ranked first.

This only covers 8 hand-written questions against the original 5 insurers (AAMI, Allianz, NRMA, RAA, RealInsurance). The corpus has since grown to 13 insurers and no ground truth exists yet for the other 8, so this number says nothing about retrieval quality on most of the current corpus. See [Limitations](#limitations-and-next-steps).

## Architecture

```mermaid
flowchart TD
    A[Claim narrative] --> B[extract_fields]
    B --> B1[LoRA r=4 adapter]
    B1 -- JSON parses and validates --> D[ClaimExtraction]
    B1 -- malformed JSON or schema failure --> B2[Claude zero-shot fallback]
    B2 --> D
    D --> C[Agent tool loop]
    C --> E[retrieve_policy]
    E --> F[(BM25 + TF-IDF index over policy chunks)]
    F --> E
    E --> C
    C --> G[get_policy_certificate]
    G --> H[(policy_certificates.json)]
    H --> G
    C --> I[calculate_payout]
    I --> C
    C --> J{Decision}
    J --> K[Approve / Deny / Escalate + citations]
```

- **Retrieval** ([code/rag/](code/rag/)): hybrid BM25 and TF-IDF search with reciprocal rank fusion over policy chunks, filtered by insurer and product when named. Grounded answer generation with page-level citations lives in [code/rag/generate.py](code/rag/generate.py).
- **Extraction** ([code/extract/extract_fields.py](code/extract/extract_fields.py)): tries the LoRA r=4 adapter first ([code/extract/lora_infer.py](code/extract/lora_infer.py)). If the output fails to parse as JSON or fails Pydantic validation, falls back to a zero-shot Claude call ([code/extract/zero_shot.py](code/extract/zero_shot.py)). The schema both share is [code/extract/schema.py](code/extract/schema.py).
- **Agent loop** ([code/agent/loop.py](code/agent/loop.py)): a tool-calling loop with a capped number of turns and a named escalation policy. Tools are defined in [code/agent/tools.py](code/agent/tools.py): `extract_fields`, `retrieve_policy`, `get_policy_certificate`, `calculate_payout`, `escalate_to_human`.
- **Payout calculation**: `calculate_payout` is plain Python arithmetic (claim capped at the coverage limit, minus excess, never negative). Its excess and coverage-limit inputs are meant to come from `get_policy_certificate`, a lookup against [data/claims/policy_certificates.json](data/claims/policy_certificates.json) (synthetic per-insurer figures standing in for real certificates). The model passes those numbers into `calculate_payout` as tool arguments, so the system prompt enforces the rule rather than the code; every logged payout so far used the looked-up figures.
- **Citations**: the agent must cite each source as `[Insurer Product DocType, page N]`. Only retrieved chunks whose tag appears in the final reasoning are kept as the decision's `citations` (each with its `doc_id`); everything retrieved is logged separately as `retrieved_sources`.
- **Batch runner** ([code/agent/run_batch.py](code/agent/run_batch.py)): runs a JSONL file of claims through the loop, writes one decision log per claim to `outputs/decisions/` (or `--out-dir`), a summary CSV, and the cost/latency report.

## Key design decisions

### Why LoRA first despite a lower score

Zero-shot Claude scores higher (0.930 vs. 0.822), but the LoRA adapter has zero marginal cost per call once trained and only fails to produce valid JSON on 2 of 54 test examples. The Claude fallback exists specifically to catch that failure mode, so the common case is free and the rare failure case still gets Claude's higher accuracy. See [outputs/phase2_findings.md](outputs/phase2_findings.md).

### Why r=4 despite a lower score

An earlier run on a smaller training set (150 examples) found r=4 clearly outperforming r=8 and r=16. After the training set grew to 207 examples, that result did not hold: all three ranks converged to statistically indistinguishable scores. r=4 is still used in production, not because it wins, but because it is the cheapest of three tied options (fewest trainable parameters, fastest to train). See [extract_fields.py](code/extract/extract_fields.py).

### Why payouts only use tool-retrieved certificate data

Excess and coverage-limit figures are policyholder-specific and never appear in PDS or KFS text, only on an individual's certificate of insurance (confirmed directly in AAMI's PDS). Before `policy_certificates.json` existed, the agent correctly refused to guess these numbers and escalated 19 of 20 batch claims. Adding a certificate lookup tool, rather than letting the model estimate a plausible-looking number, is what produces a real decision spread instead of near-universal escalation.

### Extraction metric

The per-field score in the results table is not plain accuracy. It blends three scoring rules by field type ([code/extract/eval.py](code/extract/eval.py)):
- Exact match for `claim_type`, `date_of_loss`, `policy_number`.
- 5% relative tolerance for `estimated_amount`.
- Token-set F1 for the two free-text fields, `cause` and `damaged_item`.

The six field scores are averaged into one number per example. A prediction that fails to parse or fails schema validation scores 0 on every field.

## Example decision

Input narrative (from `data/claims/batch20.jsonl`, claim-003):

> On 3 February 2023 heavy rain caused the creek behind our property to overflow and water came into our home, ruining the carpets in the living room and soaking the bottom of the plasterboard walls. It happened really fast, within about an hour the water was ankle deep inside. We've had to rip up the carpet already because it started smelling. I think it'll cost around $9,000 to fix the flooring and walls.

Tool calls made by the agent, in order:
1. `extract_fields` on the narrative text.
2. `retrieve_policy` (`"Flood cover creek overflow flash flooding damage"`, insurer: RAA).
3. `get_policy_certificate` (insurer: RAA) → excess $500, coverage limit $45,000.
4. `calculate_payout` (claim_amount: 9000, excess: 500, coverage_limit: 45000) → $8,500.

Decision (`outputs/decisions/claim-003.json`), quoted exactly as generated:

> DECISION: approve
>
> The claim describes flood damage caused by a nearby creek overflowing after heavy rain, which inundated the home and damaged carpets and the base of plasterboard walls. Under RAA's policy, "Flood" is explicitly defined to include water escaping the normal confines of a creek [RAA Contents PDS, page 87], and loss or damage resulting from Flood is covered [RAA Contents PDS, page 50]. None of the listed exclusions (unfinished structures, tennis courts, wharfs/jetties, pool/tank water, paint-only damage, hydrostatic/underground water, glasshouses, unsealed paths/driveways, malicious dam damage) apply to this scenario — the damage is structural/material (carpet and plasterboard) from surface floodwater entering the home [RAA Contents PDS, page 50]. The mixed-water-source exclusion for Actions of the Sea also does not apply here since this is a pure creek-flood event [RAA Contents PDS, page 88].
>
> Using the policyholder's certificate figures (excess: $500, coverage limit: $45,000), and the claimed amount of $9,000, the calculated payout is:
>
> **$8,500** (i.e., $9,000 claim − $500 excess), well within the $45,000 coverage limit.
>
> Citations: [RAA Contents PDS, page 87], [RAA Contents PDS, page 50], [RAA Contents PDS, page 88]

Structured `citations` in the decision log, resolved from those tags: `raa-contents-pds` pages 87, 50 and 88.


## Limitations and next steps

- **The extraction test set is synthetic (n=54).** Both the training and test narratives were generated by Claude and, for the zero-shot comparison, partly graded by Claude. A separate real-world set of 11 claims, sourced from published AFCA (Australian Financial Complaints Authority) determinations, exists in `data/claims/afca_test.jsonl` and shows the "zero-shot beats LoRA" gap does not clearly hold at that sample size. Next step: grow that set past 11 before treating the synthetic result as production-representative.
- **Retrieval ground truth only covers 5 of 13 insurers.** The corpus grew from 18 to 44 documents this project, but the hand-verified hit-rate/MRR ground truth was never extended to the 8 newer insurers. Next step: write ground truth questions for the new insurers the same way the original 8 were built, by hand-checking citations against the source PDF.
- **Citations are document and page level, not quote level.** A citation says which PDS and page a claim came from, not the exact sentence. Only one citation has been manually verified against the source PDF text; there is no systematic audit across the corpus. Next step: sample a batch of citations across many claims and check each against the source text, then report a groundedness rate.
- **A known retrieval gap exists for insurers with combined Building and Contents documents** (NRMA, Suncorp, Allianz). Filtering by `product="Contents"` for these insurers misses their combined PDS, which is tagged `product="Home"`. Next step: either multi-label combined documents or fall back to an unfiltered search when a strict product filter returns thin results.
- **Agent decisions are not deterministic between runs, and it matters.** `claude-sonnet-5` accepts no `temperature`, `top_p`, or `seed` parameter, so there is no sampling control. Measured over 4 runs of batch20: 8 of 20 claims changed decision at least once, mostly between approve and escalate (claim-010 alternated between deny and escalate). Next step: tighten the escalation rules in the system prompt, or take a majority vote over several runs at proportionally higher cost.
- **Escalation is high and its correctness is unmeasured.** 12 to 16 of 20 claims escalate per run, including routine-looking ones (burst hoses, graffiti), not just the 5 claims written to be unresolvable (015 to 019). Next step: fill in the blind gold labels and run `python -m eval.decision_agreement` to get agreement and escalation precision/recall.
- **Prompt-injection and fairness red-teaming are stubbed, not implemented** (`code/redteam/injection_suite.py`, `code/redteam/fairness_pairs.py`). Function signatures and TODOs exist; no test cases or logic have been written yet.
- **The free-text field gap (`cause`, `damaged_item`) between zero-shot and LoRA is unexplained.** It is not a training-data-volume problem (more data did not close it) and not a model-capacity problem (higher LoRA rank did not close it either). The actual cause has not been identified.

## Project status by phase

| Phase | Status |
|---|---|
| 1. Retrieval over real policy documents | Done |
| 2. Structured extraction, LoRA fine-tuning ablation | Done, with the limitations above |
| 3. Agent orchestration (tools, escalation policy, decision logs) | Done |
| 4. Cost/latency instrumentation | Done |
| 4. Prompt-injection and fairness red-teaming | Stubbed only, not implemented |
| 5. Golden-set evaluation, naive baseline comparison | In progress: stability measured, gold-label sheet and scorer built, labels not yet filled in |
| 6. Vector DB, multi-agent split, web UI | Not started |

## Setup and running it

### Requirements

- Python 3.14.
- An `ANTHROPIC_API_KEY` (required, used by every zero-shot and agent call).
- Optionally `ANTHROPIC_WORKSPACE_ID`, if your API key is workspace-scoped.
- Optionally `LLM_MODEL` to override the default model string.

Put these in a `.env` file at the repo root. Never commit real key values.

Install dependencies:

```
pip install -r requirements.txt
```

### Running the pipeline

All commands below are run from the `code/` directory.

Rebuild the policy document corpus from scratch (see [Data](#data)):
```
python -m ingest.discover_sources
python -m ingest.download
python -m ingest.extract_text
python -m rag.build_index
```

Ask a coverage question:
```
python cli.py "is storm damage to my roof covered?" --insurer AAMI --debug
```

Run structured extraction on a claim narrative:
```
python -c "from extract.extract_fields import extract_fields; print(extract_fields('your claim text here'))"
```

Run the agent on a batch of claims (`--out-dir` keeps repeat runs separate):
```
python -m agent.run_batch ../data/claims/batch20.jsonl
python -m agent.run_batch ../data/claims/batch20.jsonl --out-dir ../outputs/stability/run1
```

Score decisions against gold labels and across runs:
```
python -m eval.decision_agreement ../outputs/decisions ../outputs/stability/run1/decisions ...
```

The LoRA model runs on Apple GPU, CUDA, or CPU, whichever is available; override with `DEVICE=cpu`.

Evaluation scripts:
```
python -m extract.zero_shot --dataset test        # zero-shot baseline
python -m extract.lora_infer --rank 4 --dataset test   # LoRA adapter eval
python -m eval.bootstrap_ci ../outputs/zero_shot_results.json ../outputs/lora_r4_results.json
python -m eval.check_leakage                       # train/val/test split leakage check
python -m eval.hit_rate                            # retrieval hit-rate@5 / MRR@5
```

### Data

`data/raw_pdfs/`, `data/extracted/`, and `data/chunks.jsonl` are not committed to this repository. `data/manifest.json` and `data/sources.json` are, since they're just corpus metadata, not the documents themselves. Rebuild the actual corpus locally with the ingest commands above; every source is a public insurer disclosure page.

The ingest pipeline downloads from insurer disclosure pages whose URLs may change without notice. It was last verified end to end on 2026-09-20.

## Repo structure

```
code/
  ingest/      discover, download, and extract text from insurer PDS/KFS PDFs
  rag/         chunking, hybrid BM25+TF-IDF indexing, grounded generation with citations
  extract/     schema, synthetic data generation, zero-shot baseline, LoRA training and inference
  agent/       tool definitions, decision loop, batch runner
  eval/        bootstrap confidence intervals, leakage check, retrieval hit-rate, cost/latency instrumentation
  redteam/     prompt-injection and fairness test scaffolding (not yet implemented)
data/
  raw_pdfs/    source insurer PDF documents
  extracted/   per-page extracted text
  claims/      synthetic and real-world (AFCA) claim narratives, splits, policy certificate data
  chunks.jsonl, manifest.json, sources.json   retrieval index inputs and corpus metadata
outputs/
  decisions/           per-claim agent decision logs
  stability/           three repeat batch20 runs (decisions, summary, cost/latency report each)
  phase2_findings.md, phase3_findings.md, phase4_report.md   write-ups and measured results
  *_results.json       per-example predictions and scores for each eval run
PROJECT_LOG.md   a running, dated log of what was built, why, and what broke, phase by phase
```
