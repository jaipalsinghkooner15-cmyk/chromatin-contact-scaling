#!/usr/bin/env python3
"""
Fig 5: do DNA double-strand breaks change how contacts fall off with distance?

The data are Hi-C from DIvA cells (E-MTAB-8851, Arnould et al. 2021). Adding
OHT makes the AsiSI enzyme cut the genome at its target sites, so each pair of
files is the same cells without breaks (DIvA) and with breaks (OHT). There are
four pairs: two biological replicates (A and B), and an siRNA experiment with a
control siRNA (siControl) and cohesin knocked down (siSCC1).

Method. Every file goes through exactly the same steps as the 171 samples in
the main analysis (reading, counting and fitting all come from
preprocess/02_compute_alpha.py). Each file is read one chromosome at a time,
so we get one slope (alpha) per chromosome as well as one for the whole genome
(the whole genome is the sum of the 22 chromosomes, so it matches the main
method exactly). Alpha is shown negative, as in the rest of the paper: more
negative means contacts fall off faster with distance.

What each panel asks, and the test used:

    A  Where along the chromosome do breaks change contacts?
       The whole-genome curve with breaks divided by the curve without
       (log2 ratio), after dividing each curve by its total contacts so that
       sequencing depth cannot shift it up or down. Flat at 0 means no
       change. Descriptive, no test. (Depth does not affect alpha, because a
       slope does not change when a curve is moved up or down.)

    B  Do breaks change the slope?
       For each pair, delta alpha (OHT minus DIvA) on each of the 22
       chromosomes. We report the median delta, a 95% bootstrap confidence
       interval (resampling chromosomes), and a Wilcoxon signed-rank test,
       which asks whether the chromosome-by-chromosome changes lean one way
       more than chance would allow. It uses the ranks, so one odd chromosome
       cannot drive the result.

    C  Does knocking down cohesin (SCC1) change the effect of breaks?
       Delta alpha in siSCC1 against delta alpha in siControl, paired by
       chromosome, with the same Wilcoxon test.

The five tests (four in B, one in C) are Bonferroni corrected.

What the tests can and cannot say. The 22 chromosomes come from one
experiment, so a small p-value means the change is consistent across the
genome, not that it would repeat in a new experiment. Reproducibility comes
from replicates A and B agreeing in direction.

Inputs
    data/fig5_inputs.csv            one row per .hic file: sample_id, pair, condition, file, genome
    --search folder                 where the .hic files are
Outputs (in <data-root>/fig5)
    counts/<sample_id>.csv          contacts and bin pairs per chromosome and distance (cache)
    fig5_alpha.csv                  alpha for every file, per chromosome and whole genome
    fig5_stats.csv                  the five tests
    fig5.png, fig5.pdf              the figure

Usage
    python analysis/fig5_dna_damage.py --search "C:/.../Raw_data_HiC/dna_damage_analysis"
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # after use("Agg"): draw to files, no window

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_PATH = REPO_ROOT / "data" / "fig5_inputs.csv"
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"
PAIRS = ["Replicate A", "Replicate B", "siControl", "siSCC1"]
CONDITIONS = ["DIvA", "OHT"]
BOOTSTRAP_RESAMPLES = 10_000
SEED = 20261107

# Colours (checked for colour-blind separation): DIvA blue, OHT orange, one hue per pair.
DIVA_COLOUR, OHT_COLOUR = "#2a78d6", "#eb6834"
PAIR_COLOURS = dict(zip(PAIRS, ["#1baf7a", "#008300", "#eda100", "#e87ba4"]))
INK, MUTED, GRID, LINK = "#1f1e1b", "#6f6e66", "#e6e5df", "#cfcec6"


def load_pipeline():
    """The reading and fitting code of 02_compute_alpha.py, so the method is identical."""
    path = REPO_ROOT / "preprocess" / "02_compute_alpha.py"
    spec = importlib.util.spec_from_file_location("compute_alpha", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Counting ───────────────────────────────────────────────────────────────────

def count_by_chromosome(path: Path, pipeline) -> pd.DataFrame:
    """Contacts and bin pairs at each distance, separately for each autosome."""
    frames = []
    for chrom in pipeline.AUTOSOMES:
        totals = pipeline.DistanceTotals()
        pipeline.read_hictk_file(path, totals, only_chrom=chrom)
        if totals.bin_pairs.sum() == 0:
            raise ValueError(f"{path.name}: chromosome {chrom} not found")
        frames.append(pd.DataFrame({
            "chromosome": chrom,
            "diagonal": np.arange(pipeline.MAX_DIAGONAL + 1),
            "contacts": totals.contacts,
            "bin_pairs": totals.bin_pairs,
        }))
    return pd.concat(frames, ignore_index=True)


def load_counts(row: pd.Series, search: list[Path], out_dir: Path, force: bool,
                pipeline) -> pd.DataFrame:
    """Counts for one file, from the cache if it has been read before."""
    cache = out_dir / "counts" / f"{row['sample_id']}.csv"
    if cache.exists() and not force:
        print(f"  {row['sample_id']:16} cached")
        return pd.read_csv(cache, dtype={"chromosome": str})
    path = pipeline.locate(row["file"], pipeline.index_files(search))
    start = time.time()
    print(f"  {row['sample_id']:16} reading {path.name} ...", end=" ", flush=True)
    counts = count_by_chromosome(path, pipeline)
    cache.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(cache, index=False)
    print(f"done in {time.time() - start:.0f} s")
    return counts


# ── Alpha ──────────────────────────────────────────────────────────────────────

def curve_of(counts: pd.DataFrame, pipeline) -> pd.DataFrame:
    return pipeline.scaling_curve(counts["diagonal"].to_numpy(), counts["contacts"].to_numpy(),
                                  counts["bin_pairs"].to_numpy())


def genome_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """The 22 chromosomes summed: the same totals the main method uses."""
    return counts.groupby("diagonal", as_index=False)[["contacts", "bin_pairs"]].sum()


def alphas_for_file(row: pd.Series, counts: pd.DataFrame, pipeline) -> list[dict]:
    """Alpha for each chromosome and for the whole genome, stored negative."""
    pieces = [(chrom, part) for chrom, part in counts.groupby("chromosome", sort=False)]
    pieces.append(("genome", genome_counts(counts)))
    rows = []
    for chrom, part in pieces:
        fit = pipeline.fit_alpha(curve_of(part, pipeline))
        rows.append({
            "sample_id": row["sample_id"], "pair": row["pair"], "condition": row["condition"],
            "chromosome": chrom, "alpha": -fit["alpha"], "r_squared": fit["r_squared"],
            "n_points": fit["n_points"], "cis_contacts": int(part["contacts"].sum()),
        })
    return rows


# ── Statistics ─────────────────────────────────────────────────────────────────

def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval for the median, resampling chromosomes with replacement."""
    resamples = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True)
    low, high = np.percentile(np.median(resamples, axis=1), [2.5, 97.5])
    return float(low), float(high)


def paired_test(name: str, first: pd.Series, second: pd.Series, what: str,
                rng: np.random.Generator) -> dict:
    """Median change from first to second over matched chromosomes, CI and Wilcoxon p."""
    change = (second - first).dropna().to_numpy()
    low, high = bootstrap_median_ci(change, rng)
    return {
        "test": name, "compares": what, "n_chromosomes": len(change),
        "median_change": float(np.median(change)), "ci_low": low, "ci_high": high,
        "chromosomes_down": int((change < 0).sum()), "chromosomes_up": int((change > 0).sum()),
        "wilcoxon_p": float(stats.wilcoxon(change).pvalue),
    }


def run_stats(alphas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delta alpha per chromosome for each pair, and the five tests."""
    rng = np.random.default_rng(SEED)
    per_chrom = alphas[alphas["chromosome"] != "genome"].pivot_table(
        index="chromosome", columns=["pair", "condition"], values="alpha")
    deltas = pd.DataFrame({pair: per_chrom[(pair, "OHT")] - per_chrom[(pair, "DIvA")]
                           for pair in PAIRS})
    genome = alphas[alphas["chromosome"] == "genome"].set_index(["pair", "condition"])["alpha"]

    tests = []
    for pair in PAIRS:
        result = paired_test(f"B: {pair}", per_chrom[(pair, "DIvA")], per_chrom[(pair, "OHT")],
                             "alpha OHT minus DIvA", rng)
        result["genome_delta"] = genome[(pair, "OHT")] - genome[(pair, "DIvA")]
        tests.append(result)
    result = paired_test("C: cohesin", deltas["siControl"], deltas["siSCC1"],
                         "delta alpha siSCC1 minus siControl", rng)
    result["genome_delta"] = ((genome[("siSCC1", "OHT")] - genome[("siSCC1", "DIvA")])
                              - (genome[("siControl", "OHT")] - genome[("siControl", "DIvA")]))
    tests.append(result)

    table = pd.DataFrame(tests)
    table["wilcoxon_p_bonferroni"] = np.minimum(table["wilcoxon_p"] * len(table), 1.0)
    return deltas, table


def ratio_curves(counts: dict[str, pd.DataFrame], inputs: pd.DataFrame,
                 pipeline) -> dict[str, pd.DataFrame]:
    """log2 of P(s) with breaks over P(s) without, whole genome, for each pair.

    Each curve is first divided by its total contacts, so a file that was
    simply sequenced deeper does not sit higher. Only the shape is compared.
    """
    def depth_corrected(curve: pd.DataFrame) -> pd.Series:
        return curve["contact_probability"] / curve["contacts"].sum()

    curves = {}
    for pair in PAIRS:
        ids = inputs[inputs["pair"] == pair].set_index("condition")["sample_id"]
        diva = curve_of(genome_counts(counts[ids["DIvA"]]), pipeline)
        oht = curve_of(genome_counts(counts[ids["OHT"]]), pipeline)
        if not np.allclose(diva["distance_bp"], oht["distance_bp"]):
            raise ValueError(f"{pair}: the two files have different distance bins")
        usable = (diva["contacts"] > 0) & (oht["contacts"] > 0)
        curves[pair] = pd.DataFrame({
            "distance_bp": diva.loc[usable, "distance_bp"],
            "log2_ratio": np.log2(depth_corrected(oht)[usable] / depth_corrected(diva)[usable]),
        })
    return curves


# ── Figure ─────────────────────────────────────────────────────────────────────

def format_p(p: float) -> str:
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def paired_dots(ax, first: pd.Series, second: pd.Series, colours: tuple[str, str]) -> None:
    """One grey line per chromosome from first to second, median in black."""
    for a, b in zip(first, second):
        ax.plot([0, 1], [a, b], color=LINK, lw=0.9, zorder=1)
    for x, values, colour in ((0, first, colours[0]), (1, second, colours[1])):
        ax.scatter(np.full(len(values), x), values, s=16, color=colour, zorder=2,
                   edgecolor="white", linewidth=0.6)
    ax.plot([0, 1], [np.median(first), np.median(second)], color=INK, lw=2.2, zorder=3)
    ax.set_xlim(-0.35, 1.35)


def stat_label(result: pd.Series) -> str:
    return (f"\u0394\u03b1 = {result['median_change']:+.3f} "
            f"[{result['ci_low']:+.3f} to {result['ci_high']:+.3f}]\n"
            f"p = {format_p(result['wilcoxon_p_bonferroni'])} (Bonferroni)")


def draw_figure(alphas: pd.DataFrame, deltas: pd.DataFrame, tests: pd.DataFrame,
                ratios: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.edgecolor": "#8a897f", "axes.labelcolor": "#3d3c37",
                         "xtick.color": "#5f5e57", "ytick.color": "#5f5e57"})
    fig = plt.figure(figsize=(12, 9.6))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.1, 1.0], hspace=0.95, wspace=0.35)
    tests = tests.set_index("test")

    # A: ratio curves
    ax = fig.add_subplot(grid[0, :3])
    ax.axvspan(1e5, 1e7, color="#f1f0eb", zorder=0)
    ax.axhline(0, color="#9a998f", lw=1)
    ends = {}
    for pair in PAIRS:
        curve = ratios[pair]
        ax.semilogx(curve["distance_bp"], curve["log2_ratio"], color=PAIR_COLOURS[pair], lw=2)
        ends[pair] = curve["log2_ratio"].iloc[-1]
    span = np.ptp(np.concatenate([r["log2_ratio"] for r in ratios.values()]))
    placed: list[float] = []
    for pair in sorted(ends, key=ends.get):
        y = ends[pair] if not placed else max(ends[pair], placed[-1] + 0.09 * span)
        placed.append(y)
        ax.text(2.3e7, y, pair, color=INK, fontsize=8.5, va="center")
    ax.set_xlim(2e4, 2e7)
    ax.set_xlabel("distance s (bp)")
    ax.set_ylabel("log2 P(s) ratio\n(with breaks / without)")
    ax.text(1.1e5, 0.03, "fit range 0.1 to 10 Mb", transform=ax.get_xaxis_transform(),
            fontsize=8, color=MUTED, va="bottom")
    tidy(ax)
    note = fig.add_subplot(grid[0, 3])
    note.axis("off")
    note.text(0.25, 0.95, "Reading panel A", weight="bold", fontsize=9, color=INK, va="top")
    note.text(0.25, 0.78, "Flat at 0: breaks change\nnothing at that distance.\n"
              "Above 0: more contacts\nwith breaks.\nCorrected for depth.\n"
              "Whole genome, all 22\nautosomes.",
              fontsize=8.5, color="#3d3c37", va="top", linespacing=1.4)
    fig.text(0.07, 0.905, "A   Where along the chromosome breaks change contacts",
             fontsize=11.5, weight="bold", color=INK)

    # B: per-chromosome slopes, one panel per pair
    per_chrom = alphas[alphas["chromosome"] != "genome"].pivot_table(
        index="chromosome", columns=["pair", "condition"], values="alpha")
    shared = None
    for i, pair in enumerate(PAIRS):
        ax = fig.add_subplot(grid[1, i], sharey=shared)
        shared = shared or ax
        paired_dots(ax, per_chrom[(pair, "DIvA")], per_chrom[(pair, "OHT")],
                    (DIVA_COLOUR, OHT_COLOUR))
        ax.set_xticks([0, 1], CONDITIONS)
        ax.set_title(f"{pair}\n{stat_label(tests.loc[f'B: {pair}'])}", loc="left",
                     fontsize=8.3, color=INK, linespacing=1.35)
        if i == 0:
            ax.set_ylabel("slope \u03b1, one dot per chromosome")
        tidy(ax)
    fig.text(0.07, 0.635, "B   Do breaks change the slope?  22 autosomes, paired DIvA to OHT "
             "(black line = median)", fontsize=11.5, weight="bold", color=INK)

    # C: cohesin
    ax = fig.add_subplot(grid[2, :2])
    paired_dots(ax, deltas["siControl"], deltas["siSCC1"],
                (PAIR_COLOURS["siControl"], PAIR_COLOURS["siSCC1"]))
    ax.axhline(0, color="#9a998f", lw=1)
    ax.set_xticks([0, 1], ["siControl", "siSCC1"])
    ax.set_ylabel("\u0394\u03b1 (OHT minus DIvA)\nper chromosome")
    ax.set_title(stat_label(tests.loc["C: cohesin"]), loc="left", fontsize=8.3, color=INK,
                 linespacing=1.35)
    tidy(ax)
    fig.text(0.07, 0.305, "C   Does knocking down cohesin (SCC1) change the effect of breaks?",
             fontsize=11.5, weight="bold", color=INK)
    note = fig.add_subplot(grid[2, 2:])
    note.axis("off")
    note.text(0.02, 0.95, "What the tests can and cannot say", weight="bold", fontsize=9,
              color=INK, va="top")
    note.text(0.02, 0.80,
              "Wilcoxon signed-rank across the 22 chromosomes\nof one experiment: is the change "
              "consistent\nacross the genome? Chromosomes are not separate\nexperiments, so "
              "reproducibility comes from\nReplicates A and B agreeing in direction.\n\n"
              "\u0394\u03b1: median over chromosomes, 95% bootstrap\ninterval in brackets. "
              "Bonferroni over 5 tests.",
              fontsize=8.5, color="#3d3c37", va="top", linespacing=1.4)

    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"fig5.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Run ────────────────────────────────────────────────────────────────────────

def load_inputs(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str)
    expected = {(p, c) for p in PAIRS for c in CONDITIONS}
    found = set(zip(table["pair"], table["condition"]))
    if found != expected or len(table) != len(expected):
        raise SystemExit(f"{path.name} must have exactly one file for each pair and condition")
    return table


def print_report(alphas: pd.DataFrame, tests: pd.DataFrame) -> None:
    genome = alphas[alphas["chromosome"] == "genome"]
    print("\nWhole-genome alpha (same method as the main analysis)")
    print(genome[["sample_id", "alpha", "r_squared", "cis_contacts"]].round(4).to_string(index=False))
    print("\nTests (change = OHT minus DIvA; negative = steeper with breaks)")
    columns = ["test", "median_change", "ci_low", "ci_high", "chromosomes_down", "chromosomes_up",
               "genome_delta", "wilcoxon_p", "wilcoxon_p_bonferroni"]
    shown = tests[columns].copy()
    for column in ("wilcoxon_p", "wilcoxon_p_bonferroni"):
        shown[column] = shown[column].map(format_p)
    print(shown.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--search", type=Path, action="append", required=True,
                        help="folder with the .hic files (repeat for several folders)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--inputs", type=Path, default=INPUTS_PATH)
    parser.add_argument("--force", action="store_true", help="read the .hic files again")
    arguments = parser.parse_args()

    pipeline = load_pipeline()
    out_dir = arguments.data_root.expanduser().resolve() / "fig5"
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(arguments.inputs)

    print("Counting contacts, one chromosome at a time")
    counts = {row["sample_id"]: load_counts(row, arguments.search, out_dir, arguments.force,
                                            pipeline)
              for _, row in inputs.iterrows()}
    alphas = pd.DataFrame([r for _, row in inputs.iterrows()
                           for r in alphas_for_file(row, counts[row["sample_id"]], pipeline)])
    deltas, tests = run_stats(alphas)
    ratios = ratio_curves(counts, inputs, pipeline)

    alphas.to_csv(out_dir / "fig5_alpha.csv", index=False)
    tests.to_csv(out_dir / "fig5_stats.csv", index=False)
    draw_figure(alphas, deltas, tests, ratios, out_dir)
    print_report(alphas, tests)
    for name in ("fig5_alpha.csv", "fig5_stats.csv", "fig5.png", "fig5.pdf"):
        print(f"Saved: {out_dir / name}")


if __name__ == "__main__":
    main()
