"""Phase 5: agent decisions vs. hand-labelled gold, and run-to-run stability.

Gold labels live in data/claims/batch20_gold.csv (gold_decision column, filled
in by hand, blind to the agent's output). Each run is a directory of
claim-*.json decision files as written by agent.run_batch.

Run (from code/):
  python -m eval.decision_agreement ../outputs/decisions                      # agreement only
  python -m eval.decision_agreement ../outputs/stability/run1 ../outputs/stability/run2 ...
  python -m eval.decision_agreement --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "claims" / "batch20_gold.csv"
LABELS = ("approve", "deny", "escalate")


def load_gold(path: Path) -> dict[str, str]:
    """claim_id -> gold decision, skipping rows not labelled yet."""
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        gold = {r["claim_id"]: r["gold_decision"].strip().lower() for r in rows}
    bad = {cid: g for cid, g in gold.items() if g and g not in LABELS}
    if bad:
        raise ValueError(f"gold_decision must be one of {LABELS}: {bad}")
    return {cid: g for cid, g in gold.items() if g}


def load_run(run_dir: Path) -> dict[str, str]:
    """claim_id -> final_decision for one batch run."""
    return {
        p.stem: json.loads(p.read_text())["final_decision"]
        for p in sorted(run_dir.glob("claim-*.json"))
    }


def agreement(pred: dict[str, str], gold: dict[str, str]) -> dict:
    ids = sorted(set(pred) & set(gold))
    confusion = Counter((gold[i], pred[i]) for i in ids)
    agent_esc = sum(1 for i in ids if pred[i] == "escalate")
    gold_esc = sum(1 for i in ids if gold[i] == "escalate")
    both_esc = confusion[("escalate", "escalate")]
    return {
        "n": len(ids),
        "agree": sum(1 for i in ids if pred[i] == gold[i]),
        "confusion": confusion,
        # Of the claims the agent escalated, how many should have been?
        "escalation_precision": both_esc / agent_esc if agent_esc else None,
        # Of the claims that should be escalated, how many did the agent catch?
        "escalation_recall": both_esc / gold_esc if gold_esc else None,
        "disagreements": [(i, gold[i], pred[i]) for i in ids if pred[i] != gold[i]],
    }


def stability(runs: list[dict[str, str]]) -> dict:
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    per_claim = {i: [r[i] for r in runs] for i in ids}
    flips = {i: d for i, d in per_claim.items() if len(set(d)) > 1}
    # Majority vote; a tie falls back to escalate, the loop's own safe default.
    majority = {}
    for i, d in per_claim.items():
        (top, n), *rest = Counter(d).most_common()
        majority[i] = top if not rest or rest[0][1] < n else "escalate"
    return {"n": len(ids), "flips": flips, "majority": majority}


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def report_agreement(name: str, pred: dict[str, str], gold: dict[str, str]) -> None:
    a = agreement(pred, gold)
    print(f"{name}: {a['agree']}/{a['n']} match gold  "
          f"escalation precision={_fmt(a['escalation_precision'])} recall={_fmt(a['escalation_recall'])}")
    print("  confusion (rows=gold, cols=agent):")
    print("  " + " " * 10 + "".join(f"{p:>10}" for p in LABELS))
    for g in LABELS:
        print(f"  {g:>10}" + "".join(f"{a['confusion'][(g, p)]:>10}" for p in LABELS))
    for cid, g, p in a["disagreements"]:
        print(f"  mismatch {cid}: gold={g} agent={p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()
    if not args.run_dirs:
        parser.error("give at least one run directory")

    runs = [load_run(d) for d in args.run_dirs]
    gold = load_gold(args.gold) if args.gold.exists() else {}
    if not gold:
        print(f"No gold labels filled in yet ({args.gold}) -- skipping agreement.\n")

    for d, run in zip(args.run_dirs, runs):
        counts = Counter(run.values())
        print(f"{d}: " + ", ".join(f"{lbl}={counts[lbl]}" for lbl in LABELS))
        if gold:
            report_agreement(str(d), run, gold)

    if len(runs) > 1:
        s = stability(runs)
        print(f"\nstability over {len(runs)} runs: {s['n'] - len(s['flips'])}/{s['n']} claims unanimous")
        for cid, d in s["flips"].items():
            print(f"  flips {cid}: {' / '.join(d)}")
        counts = Counter(s["majority"].values())
        print("majority vote: " + ", ".join(f"{lbl}={counts[lbl]}" for lbl in LABELS))
        if gold:
            report_agreement("majority vote", s["majority"], gold)


def _selftest():
    gold = {"c1": "approve", "c2": "escalate", "c3": "deny", "c4": "escalate"}
    pred = {"c1": "approve", "c2": "escalate", "c3": "escalate", "c4": "approve"}
    a = agreement(pred, gold)
    assert a["agree"] == 2 and a["n"] == 4, a
    assert a["escalation_precision"] == 0.5 and a["escalation_recall"] == 0.5, a
    s = stability([
        {"c1": "approve", "c2": "deny"},
        {"c1": "approve", "c2": "escalate"},
        {"c1": "approve", "c2": "deny"},
    ])
    assert list(s["flips"]) == ["c2"] and s["majority"] == {"c1": "approve", "c2": "deny"}, s
    tie = stability([{"c1": "approve"}, {"c1": "deny"}])
    assert tie["majority"]["c1"] == "escalate", tie
    print("selftest passed")


if __name__ == "__main__":
    main()
