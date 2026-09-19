# Insurance Claims/Underwriting Triage Agent

## Context

An agent that triages incoming Australian home-insurance claims — extracting structured fields (fine-tuned model), checking coverage against real policy text (RAG), deciding via a multi-step tool-using workflow (agent), and knowing when to escalate to a human instead of guessing. Deliberately designed so RAG, prompt engineering, agents, and fine-tuning are each load-bearing rather than bolted on, with a clean, comparable eval metric at every layer.

**Iteration principle**: every phase ends in something genuinely demoable — a strict superset of the previous phase's working demo.

## Repo conventions

- Lives in `projects/insurance-claims-agent/` (sibling to `phases/`), separate from the curriculum's own `phases/19-capstone-projects/08-production-rag-chatbot/`, which stays untouched as course material.
- `data/` is gitignored at repo root already — no new gitignore entry needed.
- New deps (`peft`, `pdfplumber`) live in this project's own `code/requirements.txt`, not root.
- API keys via `.env` + `os.environ` (`ANTHROPIC_API_KEY`, `LLM_MODEL` default `"claude-sonnet-5"`).
- Reuses `chunk_text`, `BM25`, `tfidf_embed`/`cosine_similarity`, `reciprocal_rank_fusion`, `evaluate_faithfulness` from `phases/11-llm-engineering/06-rag/code/main.py` and `07-advanced-rag/code/main.py`.

## Phase 1 — RAG coverage lookup

**Skills**: document ingestion, hybrid retrieval (BM25 + TF-IDF cosine + RRF), grounded prompt construction, prompt caching.

CLI tool answering "is X covered?" against real AU home insurance PDS/KFS text, with citations. Corpus: ~20-25 documents deduped from insurers already scraped (Allianz, AAMI, Real Insurance, RAA). PDF text via `pdfplumber`. Persistence: JSON/JSONL. Generation: real Anthropic API call with `cache_control`.

## Phase 2 — Structured extraction + fine-tuning ablation

**Skills**: synthetic dataset design, structured-output prompting, LoRA fine-tuning (PEFT, local MPS training), model evaluation methodology.

Synthetic claims dataset (~100-150 labeled narratives). Baseline: zero-shot Claude extraction. Fine-tune: LoRA (not QLoRA) on Qwen2.5-1.5B-Instruct or Llama-3.2-3B-Instruct, locally on M5 Mac (16GB unified memory) via `transformers`+`peft`, `device="mps"`. Compare accuracy/F1/cost/latency. Wire winner into Phase 1's pipeline.

## Phase 3 — Agent orchestration

**Skills**: tool/function-calling design, multi-step agent planning, confidence-based escalation policy, decision tracing.

Tools: `extract_fields`, `retrieve_policy`, `calculate_payout`, `escalate_to_human`. Agent loop: plan → call tools → decide → cited explanation. Escalation on low confidence rather than guessing. Batch runner over synthetic claims producing a decision log.

## Phase 4 — LLM app engineering hardening

**Skills**: cost/latency instrumentation, prompt-injection red-teaming, fairness guardrails, audit-trail logging.

Real `cache_control` usage stats, $/claim, p95 latency. Guardrails against claim-text prompt injection and protected-attribute-referencing reasoning. Full audit trail per decision.

## Phase 5 — Evaluation & benchmark report

**Skills**: end-to-end eval design, golden-set construction, baseline/ablation methodology.

Golden set expanded to ~150-200 claims. Metrics per layer (extraction F1, retrieval precision/recall, decision accuracy, false-auto-approval rate, cost/latency). Naive baseline: single Claude call with full policy text stuffed into context, no pipeline — honest comparison either way.

## Phase 6+ — Stretch goals

Real vector DB, multi-agent split, web UI, NRMA scraper fix, larger/hosted fine-tune.

## Verification per phase

Phase 1: 5-10 hand-written coverage questions, citations checked by hand. Phase 2: ablation script + spot-check outputs. Phase 3: 20-claim batch run, review 3-5 decision logs. Phase 4: red-team suite run, spot-check audit log. Phase 5: full golden set through pipeline + baseline, sanity-check aggregate numbers.
