#!/usr/bin/env python3
"""
Depth test: does alpha change when a deep sample is thinned to tumour depth?

For each chosen sample, contacts are randomly thinned to each target depth and
alpha is refitted, many times over, with exactly the fitting code used for
every sample (scaling_curve and fit_alpha in preprocess/02_compute_alpha.py).

Thinning works on the per-distance totals saved by 02_compute_alpha.py. Keeping
each contact independently with probability p gives a binomial total at every
distance, so drawing Binomial(total, p) per distance is the same as thinning
the raw contacts and then counting them again.

Input
    <data-root>/scaling_cache/<sample_id>.csv   written by 02_compute_alpha.py
Output
    <data-root>/depth_test.csv                  one row per sample, depth and repeat

Usage
    python analysis/depth_test.py
    python analysis/depth_test.py --samples SCABER T47D A549_rep3 --repeats 20
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"

# Deep cell lines from three matched cancer types (bladder, breast, lung).
DEFAULT_SAMPLES = ["SCABER", "T47D", "A549_rep3"]
# Spans the TCGA tumours: shallowest about 5 million cis contacts, deepest about 100 million.
DEFAULT_DEPTHS = [5_000_000, 10_000_000, 30_000_000, 100_000_000]
SEED = 20261107


def load_pipeline():
    """The fitting functions of 02_compute_alpha.py, so the method is identical."""
    path = REPO_ROOT / "preprocess" / "02_compute_alpha.py"
    spec = importlib.util.spec_from_file_location("compute_alpha", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thinned_alphas(counts: pd.DataFrame, depth: int, repeats: int,
                   rng: np.random.Generator, pipeline) -> list[float]:
    """Alpha after thinning to `depth` contacts, once per repeat."""
    keep = depth / counts["contacts"].sum()
    alphas = []
    for _ in range(repeats):
        contacts = rng.binomial(counts["contacts"].to_numpy().astype(np.int64), keep)
        curve = pipeline.scaling_curve(
            counts["diagonal"].to_numpy(), contacts, counts["bin_pairs"].to_numpy()
        )
        alphas.append(pipeline.fit_alpha(curve)["alpha"])
    return alphas


def run(data_root: Path, samples: list[str], depths: list[int], repeats: int) -> pd.DataFrame:
    pipeline = load_pipeline()
    rng = np.random.default_rng(SEED)
    rows = []
    for sample in samples:
        cache = data_root / "scaling_cache" / f"{sample}.csv"
        if not cache.exists():
            raise SystemExit(f"Not found: {cache}\nRun preprocess/02_compute_alpha.py first.")
        counts = pd.read_csv(cache)
        full_depth = int(counts["contacts"].sum())
        full_alpha = pipeline.fit_alpha(pipeline.scaling_curve(
            counts["diagonal"].to_numpy(), counts["contacts"].to_numpy(),
            counts["bin_pairs"].to_numpy()))["alpha"]
        for depth in depths:
            if depth >= full_depth:
                print(f"{sample}: skipped {depth / 1e6:g} M, the sample only has "
                      f"{full_depth / 1e6:.1f} M contacts")
                continue
            for repeat, alpha in enumerate(thinned_alphas(counts, depth, repeats, rng, pipeline)):
                rows.append({
                    "sample_id": sample, "full_depth": full_depth, "full_alpha": full_alpha,
                    "depth": depth, "repeat": repeat, "alpha": alpha,
                })
    if not rows:
        raise SystemExit("No target depth is below the depth of the chosen samples.")
    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    summary = results.groupby(["sample_id", "depth"]).agg(
        full_alpha=("full_alpha", "first"),
        mean_alpha=("alpha", "mean"),
        sd_alpha=("alpha", "std"),
    ).reset_index()
    summary["shift"] = summary["mean_alpha"] - summary["full_alpha"]
    summary["depth"] = (summary["depth"] / 1e6).map("{:g} M".format)
    print("\nAlpha after thinning (mean and SD over repeats; shift = mean minus full-depth alpha)")
    print(summary.round(4).to_string(index=False))
    print(f"\nLargest absolute shift: {summary['shift'].abs().max():.4f}")
    print(f"Largest SD between repeats: {summary['sd_alpha'].max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()

    data_root = arguments.data_root.expanduser().resolve()
    results = run(data_root, arguments.samples, arguments.depths, arguments.repeats)
    print_summary(results)
    output = data_root / "depth_test.csv"
    results.to_csv(output, index=False)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
