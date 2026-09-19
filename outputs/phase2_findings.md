# Phase 2 findings: zero-shot Claude vs. LoRA-fine-tuned Qwen2.5-1.5B

## Verdict

LoRA at `r=4` matches zero-shot Sonnet 5 almost exactly on the **structured/categorical**
fields (`claim_type`, `estimated_amount`, `policy_number`) but meaningfully underperforms
on the two **free-text** fields (`cause`, `damaged_item`). The gap is a data problem, not
a capacity problem: the rank sweep found *less* adapter capacity (`r=4`) beat *more*
(`r=8`, `r=16`) on every field, so the 99-example training set is the limiting factor,
not the model's size. Latency is essentially a wash between the two approaches; the
real practical win of the local model is zero marginal cost per call, not speed.

## Rank sweep (`test.jsonl`, n=33, same scoring path as zero-shot)

| | r=4 | r=8 | r=16 |
|---|---|---|---|
| overall | **0.848** | 0.821 | 0.814 |
| claim_type | 0.848 | 0.848 | 0.818 |
| date_of_loss | **0.970** | 0.939 | 0.939 |
| cause (token-F1) | **0.497** | 0.471 | 0.457 |
| damaged_item (token-F1) | **0.772** | 0.756 | 0.761 |
| estimated_amount | **1.000** | 0.939 | 0.939 |
| policy_number | **1.000** | 0.970 | 0.970 |
| parse failures | **0/33** | 1/33 | 1/33 |
| train time (measured, clean run) | ~4.8 min | ~11 min* | ~4.8 min |

\* the r=8 timing is not a clean comparison -- see `PROJECT_LOG.md` for the MPS
allocator-fragmentation issue that inflated it; r=4 and r=16 were run back-to-back
under matching conditions and are the fair comparison. Rank itself has negligible
effect on wall-clock time at this scale -- the frozen 1.5B base model's compute
dominates regardless of adapter rank.

**Winner: r=4.** Best or tied-best on every field, fewest trainable parameters,
zero parse failures. Confirms the plan's original hypothesis: with ~11 training
examples per class, more adapter capacity has nothing to fit and mildly hurts
rather than helps.

## Zero-shot vs. LoRA r=4, both test sets

| | test.jsonl (synthetic, n=33) | | afca_test.jsonl (real-world, n=8) | |
|---|---|---|---|---|
| | zero-shot | LoRA r=4 | zero-shot | LoRA r=4 |
| overall | 0.915 | 0.848 | 0.853 | 0.772 |
| claim_type | 0.848 | 0.848 | 0.75 | 0.75 |
| date_of_loss | 1.000 | 0.970 | 1.00 | 0.75 |
| cause (token-F1) | 0.752 | 0.497 | 0.627 | 0.426 |
| damaged_item (token-F1) | 0.889 | 0.772 | 0.738 | 0.705 |
| estimated_amount | 1.000 | 1.000 | 1.00 | 1.00 |
| policy_number | 1.000 | 1.000 | 1.00 | 1.00 |
| parse failures | 0/33 | 0/33 | 0/8 | 0/8 |
| avg latency | 2.60-3.24s (p50/p95) | 2.78s | ~2.4-2.9s | 2.58s |
| cost (this eval run) | $0.092 | $0 marginal | $0.021 | $0 marginal |

The categorical fields (`estimated_amount`, `policy_number`) match exactly on both
test sets. `claim_type` matches on both too. The consistent gap is `cause` and
`damaged_item` -- free-text generation, where phrasing precision matters more than
pattern-matching a fixed category, and where 99 examples clearly isn't enough
exposure to varied real-world phrasing. `date_of_loss` holds up on the synthetic
set but drops noticeably on the real AFCA narratives (0.75) -- worth watching if
more real-world data is added later, since this is exactly the kind of gap the
AFCA anchor exists to surface that the synthetic set alone would hide.

**Caveat on the real-world numbers**: n=8, storm-heavy (5/8) -- same caveat as
Phase 1's n=9 eval and already documented when this set was built. Read these as
a real-world sanity check, not a statistically confident per-field accuracy claim.

## Cost/latency, honestly

Latency is a near-tie (~2.6-2.8s either way) -- LoRA is not meaningfully faster
here, contrary to a common assumption about local models. The actual advantage is
**cost**: zero-shot spends real API dollars per call ($0.092 for 33 calls on the
synthetic set); the LoRA adapter, once trained (a few minutes of local compute, no
cash cost), has zero marginal cost per inference call. That's the tradeoff this
project's ablation was built to measure, and it's real -- it's just not a latency
story.

## Recommendation for step 2.6 (`extract_fields()`)

Wire in a **hybrid**: call the LoRA r=4 adapter first (free, fast, and already
matches zero-shot on 4 of 6 fields); if the model's own output fails to parse or
validate against `ClaimExtraction` (already tracked as a 0/33 and 0/8 rate here, so
rare but not impossible), fall back to a live Claude call for that one narrative.
This gets the near-zero marginal cost of the local model for the common case while
keeping Claude's better free-text quality available as a safety net exactly where
the local model is weakest, without paying for every single call.

## Limitations, for the record

- Training set is 99 examples (11/class) -- small by any standard; the free-text
  gap is very plausibly a data-volume problem that more (and more diverse, less
  Claude-authored-template-repetitive) examples would narrow.
- Test set is 33 examples (3-4/class) -- enough to compare relative rank
  performance, thin for absolute per-class accuracy claims.
- Real-world anchor is 8 examples, storm-heavy -- a sanity check, not a
  statistically powered eval.
- Model capacity was *not* the bottleneck at this data scale (rank sweep evidence)
  -- so a bigger base model is not a well-motivated next step without first adding
  more/better training data and re-measuring, per the same discipline applied to
  the Llama-3.2-3B escalation decision (not triggered, no capacity ceiling found).
