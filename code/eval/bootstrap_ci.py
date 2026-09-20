"""Bootstrap confidence intervals over saved eval result files.

The rank-sweep and zero-shot comparisons all report point estimates
(overall=0.822 etc.) from a 54-example test set -- too small to trust a
difference like 0.822 vs 0.823 as real signal without knowing the noise
band around each number. Resamples the per-example scores with replacement
to get a 95% CI on the overall mean, using the existing eval.score_one
scorer so this matches exactly how the reported point estimate was computed.

Run: python -m eval.bootstrap_ci outputs/lora_r4_results.json outputs/lora_r8_results.json ...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from extract.eval import score_one

ROOT = Path(__file__).resolve().parents[2]

N_RESAMPLES = 2000


def bootstrap_ci(predictions: list[dict | None], gold: list[dict], n_resamples: int = N_RESAMPLES, seed: int = 0) -> tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high) for the overall mean-of-fields score."""
    per_example = [
        sum(score_one(p, g).values()) / len(score_one(p, g))
        for p, g in zip(predictions, gold)
    ]
    point = statistics.mean(per_example)

    rng = random.Random(seed)
    n = len(per_example)
    resample_means = []
    for _ in range(n_resamples):
        sample = [per_example[rng.randrange(n)] for _ in range(n)]
        resample_means.append(statistics.mean(sample))
    resample_means.sort()
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[int(0.975 * n_resamples) - 1]
    return point, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+", help="Paths to *_results.json files (must contain 'predictions' and 'gold')")
    args = parser.parse_args()

    for path_str in args.result_files:
        path = Path(path_str)
        data = json.loads(path.read_text())
        point, lo, hi = bootstrap_ci(data["predictions"], data["gold"])
        label = data.get("rank", data.get("dataset", path.stem))
        print(f"{path.name:30s} (n={data['n']:3d})  overall={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  (label={label})")


if __name__ == "__main__":
    main()
