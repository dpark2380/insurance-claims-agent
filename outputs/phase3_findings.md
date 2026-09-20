# Phase 3 findings: agent orchestration

## Verdict

The escalation policy works as designed: the agent refuses to guess whenever
a real input is missing (no named insurer, no resolvable coverage-limit
figure, contradictory policy text) rather than approving on incomplete
information. That correctness came at the cost of a very high escalation
rate at first (19/20 claims), which turned out to be a genuine data gap, not
an over-cautious policy -- fixing the data gap (below) produced a realistic
decision spread without loosening the policy at all.

## Decision spread (`batch20.jsonl`, n=20)

| decision | count |
|---|---|
| approve | 5 |
| deny | 1 |
| escalate | 14 |
| errors | 0 |

0 unhandled exceptions, 0 infinite loops (the 8-turn cap never had to fire).
Every claim reached a terminal decision.

## Finding: excess/coverage-limit figures don't exist in PDS text at all

The first full batch run, before any certificate data existed, escalated
19/20 claims -- not a bug. `get_policy_certificate` didn't exist yet, and
`retrieve_policy` genuinely cannot answer "what's this policyholder's excess"
from PDS/KFS text, because that figure is customer-specific and only ever
printed on an individual certificate of insurance (confirmed directly in
AAMI's own PDS: "shown on your certificate of insurance," page 17/23). The
agent was correctly refusing to invent a number that isn't retrievable from
any tool it has.

**Fix:** added `data/claims/policy_certificates.json`, synthetic
certificate-of-insurance data standing in for the missing data source, plus
a `get_policy_certificate(insurer)` tool. This unlocked a realistic decision
spread (above) without changing the escalation policy's logic at all -- the
policy was never the problem, the missing data source was.

**Caveat, disclosed prominently on purpose:** this certificate data is
entirely synthetic (mock excess ~$450-600, coverage limit ~$40k-60k per
insurer, invented to be insurer-plausible, not fitted to any real number).
Any decision-spread statistic reported from this batch reflects the shape of
this mock data as much as the agent's real behavior -- read the 5/1/14 split
as a demonstration that all three decision paths are reachable, not as a
production-representative approval rate.

**Follow-on gap this created:** when the RAG corpus was later expanded to 12
insurers, the certificate file initially still only covered the original 5 --
any claim naming one of the 7 new insurers would escalate for a
data-availability reason unrelated to genuine ambiguity. Closed by adding
matching mock entries for all 7 new insurers. Worth remembering going
forward: every new insurer added to the RAG corpus needs a matching
certificate entry, or Phase 3's escalation-rate numbers will silently drift
for reasons that have nothing to do with the escalation policy.

## Finding: decisions are not deterministic run-to-run

The same claim (`claim-000`) was observed producing different final
decisions (`approve` in one run, `escalate` in a rerun) with no code changes
in between. Attempted the standard fix -- pin `temperature=0` on the three
Claude call sites in the decision pipeline (`agent/loop.py`,
`rag/generate.py`, `extract/zero_shot.py`) -- and it broke immediately:
`TypeError: Messages.create() got an unexpected keyword argument
'temperature'`.

Inspected the installed `anthropic==1.3.0` SDK's `Messages.create()`
signature directly: **this Claude 5-family API surface exposes no
`temperature`, `top_p`, or any other sampling-control parameter at all**,
and neither `output_config` nor `thinking` provide an equivalent. This is a
genuine capability gap in the current API for this model generation, not a
mistake to work around. The fix was reverted (it broke every API call site
in the project); the non-determinism itself remains unresolved.

**Practical implication:** any two runs of `run_batch.py` on the same claims
file may produce different individual decisions, even though the aggregate
escalation-heavy pattern is stable. Treat single-run decision-spread numbers
(including the 5/1/14 split above) as one sample, not a fixed ground truth --
this is the same "don't trust a point estimate on one run" caution that
motivated the Phase 2 bootstrap CI work, just without a CI equivalent
available yet for categorical agent decisions.

## Verification performed

- **Manual single-claim trace**, read in full before the first batch run:
  confirmed sensible tool-call ordering (`extract_fields` →
  `retrieve_policy` → escalate, with a clearly stated reason) on a
  flood-damage claim with no named insurer.
- **One citation hand-checked against real PDF text**: an `approve` decision
  cited `[AAMI Contents PDS, page 44]` for flood coverage; the actual
  extracted page 44 text opens with "Flood — We cover loss or damage caused
  by flood," confirming the citation is genuinely grounded, not
  hallucinated. This is a single-anecdote check, not a systematic audit --
  see Limitations.
- **Code review** (a separate `/code-review xhigh` pass) found and fixed 3
  real bugs before this findings doc was written: a substring-match bug in
  `_parse_decision` (a reasoning sentence merely mentioning another decision
  label could be misparsed as that decision), an unhandled `KeyError` if the
  model called `escalate_to_human` without a `reason` argument, and a
  batch-crashing `KeyError` on a claims-file row missing `claim_id`. All
  three fixed and verified via the smoke tests added to `loop.py`/`tools.py`.
- **Clean 20-claim batch run** after all fixes: 0 errors, 0 unhandled
  exceptions, decision spread as reported above.

## Limitations, for the record

- **Citation groundedness has only been spot-checked once.** One verified
  citation out of thousands of possible (claim, chunk) pairs across a
  3,573-chunk corpus is evidence the happy path works, not evidence the
  property holds reliably. A systematic audit (sample ~30 citations across
  multiple claims, verify each against the source PDF) has not been done.
- **Decision non-determinism is unresolved**, not just uncharacterized (see
  above) -- there is currently no available API-level fix for this model
  generation.
- **The certificate data is synthetic** and its specific values were chosen
  to be plausible, not derived from any real data source -- any statistic
  drawn from it (approval rate, average payout) reflects that.
- **No naive baseline exists yet** to say whether the agent's ~70%
  escalation rate on this batch is "appropriately cautious" or "punting too
  much" -- that comparison is scoped to Phase 5, not built yet.
- **`batch20.jsonl`'s 5 engineered edge cases were designed to trigger
  escalation** deliberately; the 15 "clean" cases are themselves synthetic
  narratives from the same generation process critiqued in
  `phase2_findings.md` (uniform style until the register-diversity fix,
  which `batch20.jsonl` predates) -- the batch has not been regenerated
  against the more diverse 300-example pool.
