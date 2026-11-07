#!/usr/bin/env python3
"""
Statistics: do cancer cell lines fold their DNA differently from tumours?

What this is for
    Every sample has one number, alpha, that describes how quickly contacts
    between two pieces of DNA fall away as they get further apart along the
    chromosome. This script asks whether that number differs between primary
    tumours and the cancer cell lines used to model them, overall and within
    each cancer type, and writes down every number the paper reports.

Why it is built this way (the four decisions, agreed 7 Nov 2026)
    1. Each patient or cell line counts once. Repeat runs of the same cells are
       averaged first (A549 rep1 to rep4), then cultures from the same patient
       or parental line are averaged (a relapse with its primary, the centre and
       edge of one tumour, GBM39 with GBM39-ER). Counting them separately would
       treat near-copies as independent evidence and make results look stronger
       than they are. Counting them separately is kept as a check.
    2. All tumours are used in the main analysis. Most tumours were measured by
       HiChIP and most cell lines by Hi-C, so a second check uses only tumours
       measured by Hi-C, to see whether the assay could explain the result.
    3. Tumours and cell lines are compared overall, and within each cancer type
       that has at least MIN_UNITS of both; types with fewer are described but
       not tested, because a test on one or two samples means nothing. The
       within-type p-values are Bonferroni corrected for the number of types
       tested. GBM cell lines are over half of all cell lines, so the overall
       comparison is repeated with cancer type taken into account.
    4. Normal cells (normal cell lines and normal tissue) are shown as their own
       group for reference. There are too few of them to test.

The tests, in plain terms
    Mann-Whitney U   are the values in one group generally higher than in the
                     other? Uses ranks, so one extreme sample cannot drive it.
    Welch t-test     are the group averages different? Does not assume the two
                     groups vary by the same amount.
    Cohen's d        how big is the difference, in units of the groups' spread.
                     Roughly 0.2 small, 0.5 medium, 0.8 large.
    Median delta     median alpha of cell lines minus median alpha of tumours.
    Type-adjusted    the average cell line minus tumour difference within the
                     same cancer type (least squares with cancer type as a
                     factor, matched types only), with a 95% confidence interval.

Sign convention
    alpha is negative here (it is the slope of the contact curve), as in the
    paper: a more negative value means contacts fall away faster. A negative
    delta or Cohen's d therefore means cell lines decay more steeply than tumours.

Inputs
    <data-root>/master_alpha_table.csv   written by preprocess/04_build_master_alpha_table.py
Outputs
    <data-root>/analysis_units.csv       every patient or cell line used, with its alpha
    <data-root>/stats_results.csv        every comparison, with group sizes and all test results

Usage
    python analysis/compare_alpha.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"
MIN_UNITS = 3

TUMOUR = "tumour"
CELL_LINE = "cancer_cell_line"
NORMAL_CLASSES = {"normal_cell_line", "normal_primary"}
GROUP_ORDER = ["tumour", "cancer cell line", "normal"]


# ── From samples to independent units ──────────────────────────────────────────

def load_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Not found: {path}\nRun preprocess/04_build_master_alpha_table.py first.")
    table = pd.read_csv(path, keep_default_na=False, na_values={"alpha": [""]})
    if not table["reviewed"].astype(str).eq("True").all():
        raise SystemExit("Some samples are not reviewed yet; finish the annotations first.")
    table = table[table["exclude"].astype(str) != "True"]
    group = table["sample_class"].map(
        lambda c: "tumour" if c == TUMOUR else "cancer cell line" if c == CELL_LINE
        else "normal" if c in NORMAL_CLASSES else None
    )
    table = table.assign(group=group).dropna(subset=["group"])
    has_parent = table["parent_line"] != ""
    table["origin"] = table["parent_line"].where(has_parent, table["biological_unit"])
    table["hic_only"] = ~table["assay"].str.startswith("HiChIP")
    return table


def collapse(samples: pd.DataFrame, level: str) -> pd.DataFrame:
    """One row per unit: replicates averaged, then units of one origin averaged.

    level "origin" counts each patient or parental line once (main analysis);
    level "unit" counts related cultures separately (check).
    """
    shared = ["group", "cancer_type"]
    per_unit = samples.groupby("biological_unit").agg(
        origin=("origin", "first"), group=("group", "first"), cancer_type=("cancer_type", "first"),
        hic_only=("hic_only", "all"), alpha=("alpha", "mean"), n_samples=("sample_id", "size"),
        samples=("sample_id", lambda ids: "; ".join(sorted(ids))),
    ).reset_index()
    if level == "unit":
        return per_unit.rename(columns={"biological_unit": "unit"})

    disagreements = per_unit.groupby("origin")[shared].nunique()
    conflicting = disagreements[(disagreements > 1).any(axis=1)]
    if not conflicting.empty:
        raise SystemExit("Units of one origin disagree on group or cancer type:\n"
                         + conflicting.to_string())
    per_origin = per_unit.groupby("origin").agg(
        group=("group", "first"), cancer_type=("cancer_type", "first"),
        hic_only=("hic_only", "all"), alpha=("alpha", "mean"), n_samples=("n_samples", "sum"),
        samples=("samples", "; ".join),
    ).reset_index()
    return per_origin.rename(columns={"origin": "unit"})


# ── Tests ──────────────────────────────────────────────────────────────────────

def cohens_d(cell_lines: np.ndarray, tumours: np.ndarray) -> float:
    """Difference in means divided by the pooled standard deviation."""
    n1, n2 = len(cell_lines), len(tumours)
    pooled_var = (
        (n1 - 1) * cell_lines.var(ddof=1) + (n2 - 1) * tumours.var(ddof=1)
    ) / (n1 + n2 - 2)
    return (cell_lines.mean() - tumours.mean()) / np.sqrt(pooled_var)


def compare(units: pd.DataFrame, analysis: str, comparison: str) -> dict:
    """Cell lines against tumours within the given units."""
    tumours = units.loc[units["group"] == "tumour", "alpha"].to_numpy()
    cell_lines = units.loc[units["group"] == "cancer cell line", "alpha"].to_numpy()
    row = {
        "analysis": analysis, "comparison": comparison,
        "n_tumour": len(tumours), "n_cell_line": len(cell_lines),
        "median_tumour": np.median(tumours) if len(tumours) else np.nan,
        "median_cell_line": np.median(cell_lines) if len(cell_lines) else np.nan,
    }
    row["median_delta"] = row["median_cell_line"] - row["median_tumour"]
    if min(len(tumours), len(cell_lines)) < MIN_UNITS:
        row["note"] = f"not tested: fewer than {MIN_UNITS} units in a group"
        return row
    row["mann_whitney_p"] = stats.mannwhitneyu(cell_lines, tumours, alternative="two-sided").pvalue
    row["welch_p"] = stats.ttest_ind(cell_lines, tumours, equal_var=False).pvalue
    row["cohens_d"] = cohens_d(cell_lines, tumours)
    return row


def type_adjusted(units: pd.DataFrame, analysis: str) -> dict:
    """Cell line minus tumour difference within cancer type, by least squares.

    alpha = intercept + difference * (is cell line) + one shift per cancer type,
    fitted on the cancer types that have both tumours and cell lines.
    """
    both = units[units["group"].isin(["tumour", "cancer cell line"])]
    matched = both.groupby("cancer_type")["group"].nunique()
    data = both[both["cancer_type"].isin(matched[matched == 2].index)]
    types = sorted(data["cancer_type"].unique())
    design = np.column_stack(
        [np.ones(len(data)), (data["group"] == "cancer cell line").to_numpy(float)]
        + [(data["cancer_type"] == t).to_numpy(float) for t in types[1:]]
    )
    y = data["alpha"].to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual_df = len(y) - design.shape[1]
    residual_var = np.sum((y - design @ coefficients) ** 2) / residual_df
    standard_error = np.sqrt(residual_var * np.linalg.inv(design.T @ design)[1, 1])
    t_value = coefficients[1] / standard_error
    margin = stats.t.ppf(0.975, residual_df) * standard_error
    return {
        "analysis": analysis, "comparison": f"type-adjusted ({len(types)} matched types)",
        "n_tumour": int((data["group"] == "tumour").sum()),
        "n_cell_line": int((data["group"] == "cancer cell line").sum()),
        "adjusted_delta": coefficients[1],
        "ci_low": coefficients[1] - margin, "ci_high": coefficients[1] + margin,
        "adjusted_p": 2 * stats.t.sf(abs(t_value), residual_df),
    }


def run_analysis(units: pd.DataFrame, analysis: str) -> list[dict]:
    """Overall, type-adjusted and per-type comparisons, with Bonferroni correction."""
    rows = [compare(units, analysis, "overall"), type_adjusted(units, analysis)]
    per_type = [compare(units[units["cancer_type"] == t], analysis, t)
                for t in sorted(units.loc[units["group"] != "normal", "cancer_type"].unique())]
    per_type = [r for r in per_type if r["n_tumour"] and r["n_cell_line"]]
    tested = [r for r in per_type if "mann_whitney_p" in r]
    for r in tested:
        r["bonferroni_k"] = len(tested)
        r["mann_whitney_p_bonferroni"] = min(1.0, r["mann_whitney_p"] * len(tested))
    return rows + per_type


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: pd.DataFrame, units: pd.DataFrame) -> None:
    print("\nUnits used in the main analysis (each patient or cell line once)")
    print("----------------------------------------------------------------")
    counts = pd.crosstab(units["cancer_type"].replace("", "(none)"), units["group"])
    print(counts.reindex(columns=[g for g in GROUP_ORDER if g in counts.columns]).to_string())

    normals = units[units["group"] == "normal"]
    print("\nNormal reference group (described, not tested)")
    print("-----------------------------------------------")
    print(normals[["unit", "alpha", "samples"]].round(3).to_string(index=False))
    for group in GROUP_ORDER:
        values = units.loc[units["group"] == group, "alpha"]
        print(f"  median {group:16} {values.median():.3f}  (n = {len(values)})")

    columns = ["comparison", "n_tumour", "n_cell_line", "median_tumour", "median_cell_line",
               "median_delta", "mann_whitney_p", "welch_p", "cohens_d",
               "mann_whitney_p_bonferroni", "note"]
    for analysis, part in results.groupby("analysis", sort=False):
        print(f"\n{analysis}")
        print("-" * len(analysis))
        plain = part[~part["comparison"].str.startswith("type-adjusted")]
        shown = plain[[c for c in columns if c in plain.columns]].copy()
        for column in ["mann_whitney_p", "welch_p", "mann_whitney_p_bonferroni"]:
            if column in shown:
                shown[column] = shown[column].map(lambda p: "" if pd.isna(p) else f"{p:.2g}")
        print(shown.round(3).fillna("").to_string(index=False))
        adjusted = part[part["comparison"].str.startswith("type-adjusted")].iloc[0]
        print(f"  {adjusted['comparison']}: delta = {adjusted['adjusted_delta']:.3f} "
              f"(95% CI {adjusted['ci_low']:.3f} to {adjusted['ci_high']:.3f}), "
              f"p = {adjusted['adjusted_p']:.2g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    data_root = parser.parse_args().data_root.expanduser().resolve()

    samples = load_samples(data_root / "master_alpha_table.csv")
    main_units = collapse(samples, "origin")
    separate_units = collapse(samples, "unit")
    hic_units = main_units[(main_units["group"] != "tumour") | main_units["hic_only"]]

    results = pd.DataFrame(
        run_analysis(main_units, "Main: each patient or cell line once")
        + run_analysis(separate_units, "Check 1: related cultures counted separately")
        + run_analysis(hic_units, "Check 2: Hi-C tumours only")
    )
    print_report(results, main_units)

    main_units.to_csv(data_root / "analysis_units.csv", index=False)
    results.to_csv(data_root / "stats_results.csv", index=False)
    print(f"\nSaved: {data_root / 'analysis_units.csv'}")
    print(f"Saved: {data_root / 'stats_results.csv'}")


if __name__ == "__main__":
    main()
