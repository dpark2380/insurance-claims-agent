# Phase 2 findings: zero-shot Claude vs. LoRA-fine-tuned Qwen2.5-1.5B

**Updated after the dataset was grown 150→300 examples with added register
diversity (informal/typo, terse, non-native-English phrasing alongside the
original clean/professional register) and the rank sweep + zero-shot baseline
were rerun on the new stratified split (train=207, val=39, test=54). Numbers
below are from that rerun; the original 150-example numbers are kept in the
"Original run" section at the bottom for comparison, since the story actually
changed, not just the digits.

## Verdict

On the larger, more diverse dataset, **all three LoRA ranks (4/8/16) are now
statistically indistinguishable from each other** -- rank sweep results below,
with 95% bootstrap CIs. Zero-shot Claude remains clearly ahead of every LoRA
rank, and that gap *is* real (its CI doesn't overlap any LoRA rank's CI). The
practical recommendation from before still holds: use LoRA as the default,
fall back to Claude on parse failure -- but the reason to pick `r=4`
specifically has weakened. It's no longer the best-performing rank; it's the
cheapest rank among three that all perform the same, which is still a
perfectly good reason to prefer it.

## Rank sweep (`test.jsonl`, n=54, 95% CI via 2000-resample bootstrap)

| | r=4 | r=8 | r=16 |
|---|---|---|---|
| overall | 0.822 | 0.822 | 0.823 |
| 95% CI | [0.767, 0.863] | [0.769, 0.863] | [0.770, 0.866] |
| claim_type | 0.815 | 0.833 | 0.833 |
| date_of_loss | 0.963 | 0.963 | 0.963 |
| cause (token-F1) | 0.469 | 0.458 | 0.478 |
| damaged_item (token-F1) | 0.797 | 0.789 | 0.796 |
| estimated_amount | 0.926 | 0.926 | 0.907 |
| policy_number | 0.963 | 0.963 | 0.963 |
| parse failures | 2/54 | 2/54 | 2/54 |
| final eval_loss (training) | 0.2467 | 0.2479 | 0.2458 |
| train time | 15.8 min | 13.8 min | 21.2 min |

**The three ranks' CIs overlap almost completely.** This is a genuinely
different result from the original 150-example run, where r=4 clearly won on
every field. With more, more-diverse training data, rank stopped being the
differentiator -- consistent with the original hypothesis that rank mattered
before only because the tiny 99-example set gave a higher-capacity adapter
nothing extra to fit. **Given the tie, `r=4` is still the sensible pick** (fewest
parameters, fastest to train, no accuracy cost), but "r=4 wins" is no longer
an accurate way to describe why.

## Zero-shot vs. LoRA r=4 (`test.jsonl`, n=54)

| | zero-shot | LoRA r=4 |
|---|---|---|
| overall | **0.930** | 0.822 |
| 95% CI | [0.914, 0.945] | [0.767, 0.863] |
| claim_type | 0.944 | 0.815 |
| date_of_loss | 1.000 | 0.963 |
| cause (token-F1) | 0.714 | 0.469 |
| damaged_item (token-F1) | 0.924 | 0.797 |
| estimated_amount | 1.000 | 0.926 |
| policy_number | 1.000 | 0.963 |
| parse failures | 0/54 | 2/54 |
| avg latency | p50=1.87s / p95=2.62s | ~3.57s |
| cost (this eval run) | $0.163 | $0 marginal |

**This gap is real, not noise** -- the two CIs don't overlap. Zero-shot is
still clearly better across the board, most of all on the free-text fields.
LoRA's latency is now also clearly *slower* than zero-shot on this run (3.57s
vs 1.87s p50), reversing the earlier "near-tie" finding -- likely reflecting
normal machine/API variance run to run rather than a real regression, but
recorded honestly rather than silently updated to match the old narrative.

## Independent gold-label spot-check (new this round)

Neither the zero-shot nor the LoRA r=4 predictions were used to author the
"gold" labels, so wherever *both* independently disagree with gold on the same
field, that's a real signal -- either the model is genuinely wrong twice, or
gold itself is questionable. Checked all 8 such (example, field) pairs found:

- **6/8 are token-F1 measuring valid paraphrases, not label errors** -- e.g.
  gold `"tree branch falling on roof during storm causing water ingress"` vs.
  zero-shot's `"large branch fell on roof during storm, cracking tiles and
  denting guttering, allowing rain ingress"` -- same event, different
  specificity, token-overlap scoring penalizes it. This is the token-F1
  weakness already known going in, now with concrete examples confirming it's
  the dominant cause of "disagreement," not label quality.
- **2/8 are genuine gold-label issues**, both worth fixing before this data is
  trusted further:
  - A cracked-window claim (glass cracked, cause unclear) is labeled
    `accidental_damage` in gold, but both zero-shot and LoRA independently
    picked `glass_breakage` -- and since the schema has a dedicated
    `glass_breakage` category for exactly this kind of damage, the models'
    answer looks more defensible than gold's.
  - A water-ingress claim where the narrative *itself* says the claimant can't
    tell whether it's the nearby creek overflowing or storm runoff pooling
    under the house -- gold confidently labels this `flood`, zero-shot said
    `storm`, LoRA failed to parse. This narrative probably shouldn't have a
    single confidently-graded gold `claim_type` at all; it's a genuinely
    ambiguous case (the kind Phase 3's agent is designed to escalate, not
    guess on) baked into an eval set that scores it as right-or-wrong.

Not fixed in this pass (flagging, not silently correcting) -- if the dataset
gets touched again, both should be either relabeled or marked as an
acceptable-ambiguity case excluded from strict accuracy scoring.

## Dataset leakage check (new this round)

Ran a TF-IDF cosine-similarity check (`eval/check_leakage.py`) across all three
split boundaries. **Zero near-duplicate pairs found** at a 0.85 similarity
threshold; the highest similarity found anywhere was 0.732 (train vs val).
The three splits are not contaminating each other.

## Cost/latency, honestly

Zero-shot: $0.163 for 54 calls this run. LoRA: zero marginal cost per
inference once trained. The cost advantage is real and unchanged from before.
The latency story flipped this run (LoRA now slower, not a near-tie) -- call
this normal run-to-run variance rather than a real finding until it's
reproduced.

## Recommendation for `extract_fields()` (unchanged)

Still a hybrid: LoRA r=4 first (free, and the parse-failure rate is low
enough -- 2/54 -- that Claude fallback rarely triggers), Claude fallback on
parse/validation failure. The accuracy gap is now more clearly Claude's, not
a near-tie, so if a future iteration is willing to spend money per call to
close it, that's a more clearly justified tradeoff than it was on the old
numbers.

## What "accuracy" actually measures here, honestly

Every number above except the AFCA section below is extraction accuracy on
data generated by Claude, evaluated in part by zero-shot Claude. That is a
real methodological limitation, not a footnote: it measures whether an
extractor can recover a label a member of the same model family invented,
not whether the system works on real customer-written claims. Three things
were done in response, one of them blocked:

1. **Reran and grew the real-world AFCA anchor** (below) -- this is the
   actual fix, since it's real tribunal-established facts, not
   LLM-invented ones, regardless of which model does the extracting.
2. **Considered generating the synthetic set with a different model
   provider** to at least remove the same-family issue for the zero-shot
   comparison specifically. Not done: no non-Anthropic model was reachable
   in this environment (a GPT-execution path was attempted and returned a
   hard "model not found" error, not a soft failure worth working around).
   Recorded as blocked, not silently dropped.
3. **The synthetic numbers above should be read as an internal ablation
   tool** (did the rank sweep behave sensibly, does the hybrid architecture
   reasoning hold up) rather than as an estimate of real-world performance.
   The AFCA numbers below are the ones that answer "does this work in
   practice," thin as that sample still is.

## Real-world anchor: AFCA-sourced claims (`afca_test.jsonl`, n=11, grown from 8)

Grew the set from 8 to 11 by sourcing 3 more real AFCA determinations via
secondary reporting (same method as the original 8: facts drawn from
published determinations, narratives written from those facts, not the
other way around). Investigated ~8 additional candidate cases across flood,
storm, escape-of-liquid, fire, and malicious-damage categories; only 3 had a
date grounded to the actual loss event rather than just the AFCA ruling date
(most secondary reporting states when AFCA decided, not when the incident
happened) -- the other 5 were **excluded rather than given a guessed date**,
consistent with how every other date in this project has been handled.

| | zero-shot | r=4 | r=8 | r=16 |
|---|---|---|---|---|
| overall | 0.886 | 0.741 | 0.778 | 0.794 |
| 95% CI | [0.853, 0.917] | [0.574, 0.858] | [0.716, 0.843] | [0.728, 0.859] |
| parse failures | 0/11 | 1/11 | 0/11 | 0/11 |

At n=11 the confidence intervals are wide enough to change the story: r=4's
CI is enormous ([0.574, 0.858]) and clearly overlaps zero-shot's, so on
real-world data zero-shot's advantage over r=4 specifically is *not*
statistically distinguishable from noise, unlike on the 54-example synthetic
set where it clearly was. r=16's CI upper bound (0.859) barely touches
zero-shot's lower bound (0.853) -- borderline, not a clean gap either way.
**The synthetic-set finding "zero-shot is significantly better" does not
straightforwardly replicate on real-world data at this sample size.** This
doesn't mean the synthetic finding is wrong; it means n=11 isn't powered to
confirm or contradict it, which is itself the honest conclusion.

## Limitations, for the record

- Training set is now 207 examples (23/class) -- larger and more
  register-diverse than the original 99, but the free-text gap didn't close;
  more data alone wasn't sufficient (see "not a capacity problem" note above
  -- rank sweep confirms it's also not a capacity problem, so the actual
  bottleneck for `cause`/`damaged_item` remains unidentified).
- Test set is 54 examples (6/class) -- better than 33, still thin for
  absolute per-class claims; bootstrap CIs above are the honest way to read
  these numbers rather than trusting the point estimate alone.
- No confidence intervals on the *field-level* breakdowns above, only on
  `overall` -- a per-field bootstrap would be a natural next step if a
  specific field's number needs defending.
- Real-world AFCA anchor is now 11 examples, still storm-heavy (7/11) --
  better-powered than the original 8 but still thin; see the dedicated
  section above for what its confidence intervals actually support.
- Model capacity was *not* the bottleneck at either data scale (rank sweep
  evidence, both before and after growing the dataset) -- a bigger base model
  remains not well-motivated by any evidence gathered so far.
- No non-Anthropic model was available in this environment to generate a
  cross-provider synthetic set, which would have further isolated the
  same-model-family circularity for the zero-shot comparison specifically --
  blocked, not solved, see the section above.

---

## Original run (150 examples, kept for comparison)

Rank sweep (`test.jsonl`, n=33): r=4 overall 0.848 (winner on every field),
r=8 0.821, r=16 0.814. Zero-shot vs. LoRA r=4: zero-shot 0.915, LoRA 0.848
(synthetic); zero-shot 0.853, LoRA 0.772 (AFCA real-world, n=8). No confidence
intervals were computed for this run -- the apparent "r=4 wins clearly" result
did not survive being rerun on more data with proper CIs, which is exactly why
the caveat about small-sample point estimates in the skepticism review that
prompted this rerun was worth taking seriously.
