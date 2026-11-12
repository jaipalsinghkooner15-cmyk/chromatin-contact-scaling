#!/usr/bin/env python3
"""
Figures 1 to 4 and supplementary S1, drawn only from the analysis tables.

What this is for
    The paper's figures must show exactly the numbers the statistics produced.
    So nothing here is typed in by hand: every alpha, n, delta and p-value on a
    figure is read from the tables written by analysis/compare_alpha.py, and
    every contact curve comes from ps_curves.csv, written in the same run as
    alpha by preprocess/02_compute_alpha.py.

    Before drawing, the numbers are checked twice over. The group sizes and
    median deltas are recounted from the units table and compared with the
    statistics table; if any disagree the script stops. Everything a figure
    prints is listed in figure_numbers.csv, so the text can be checked against
    it line by line.

The figures
    Fig 1   alpha of every tumour and cancer cell line, by cancer type
    Fig 2   A: tumour against cell line within each matched cancer type
            B: all tumours against all cancer cell lines
    Fig 3   contact curves of every tumour and cell line, and the group medians
    Fig 4   tumour against cell line curves within each matched cancer type
    Fig S1  as Fig 2A, using only tumours measured by Hi-C (check 2)

Choices that apply to every figure
    Units       Each patient or cell line counts once, as in the statistics.
    Medians     Groups are summarised by medians, matching the tests.
    Sign        alpha is negative; more negative means contacts fall away faster.
                Delta alpha is cell line minus tumour.
    Normals     Shown as two faint dashed lines: the median of normal tissue and
                primary cells, and the median of cultured normal lines. They are
                for reference only and are not tested.
    Curves      Each curve is scaled to 1 at 100 kb, the start of the fitting
                window, so curves from deeper and shallower samples start
                together and only the fall-off with distance is compared.

Inputs (in <data-root>)
    analysis_units.csv, stats_results.csv   written by analysis/compare_alpha.py
    master_alpha_table.csv                  written by preprocess/04_build_master_alpha_table.py
    ps_curves.csv                           written by preprocess/02_compute_alpha.py
Outputs (in <data-root>/figures)
    fig1 to fig4 and figS1 as .png and .pdf, and figure_numbers.csv

Usage
    python analysis/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # after use("Agg"): draw to files, no window
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"
MAIN = "Main: each patient or cell line once"
HIC_ONLY = "Check 2: Hi-C tumours only"
TUMOUR, CELL_LINE, NORMAL = "tumour", "cancer cell line", "normal"
ANCHOR_BP = 1e5                      # curves are scaled to 1 here
FIT_RANGE = (1e5, 1e7)
# Display names Bingxin asked for (PDF comments, p. 11); tables keep the TCGA codes.
TYPE_LABELS = {"GBM": "Brain (GBM)", "SKCM": "Skin"}
MIN_BOX = 3                          # fewer units than this: dots and median only
SEED = 20261110                      # only moves dots sideways (jitter)

# Colours (checked for colour-blind separation): tumour orange, cell line blue.
COLOUR = {TUMOUR: "#eb6834", CELL_LINE: "#2a78d6"}
NORMAL_LINES = {"normal tissue and primary cells": "#55544e",
                "cultured normal lines": "#9a998f"}
INK, MUTED, GRID, FIT_SHADE = "#1f1e1b", "#6f6e66", "#e6e5df", "#f1f0eb"


# ── Loading and checking ───────────────────────────────────────────────────────

def load_tables(data_root: Path) -> dict[str, pd.DataFrame]:
    names = ["analysis_units", "stats_results", "master_alpha_table", "ps_curves"]
    tables = {}
    for name in names:
        path = data_root / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        tables[name] = pd.read_csv(path)
    return tables


def unit_samples(units: pd.DataFrame) -> dict[str, list[str]]:
    return {row.unit: row.samples.split("; ") for row in units.itertuples()}


def normal_references(units: pd.DataFrame, master: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """Median alpha and n of the two kinds of normal, for the dashed reference lines."""
    sample_class = master.set_index("sample_id")["sample_class"]
    normals = units[units["group"] == NORMAL].copy()
    cultured = normals["samples"].map(
        lambda s: sample_class[s.split("; ")[0]] == "normal_cell_line")
    return {
        "normal tissue and primary cells": (normals.loc[~cultured, "alpha"].median(),
                                            int((~cultured).sum())),
        "cultured normal lines": (normals.loc[cultured, "alpha"].median(), int(cultured.sum())),
    }


def comparisons(stats_table: pd.DataFrame, analysis: str) -> pd.DataFrame:
    rows = stats_table[stats_table["analysis"] == analysis]
    return rows[~rows["comparison"].str.startswith("type-adjusted")].set_index("comparison")


def check_against_stats(units: pd.DataFrame, stats_table: pd.DataFrame) -> pd.DataFrame:
    """Recount n and median delta from the units table; stop if any disagree with the stats."""
    main_units = units[units["group"].isin([TUMOUR, CELL_LINE])]
    records = []
    for comparison, row in comparisons(stats_table, MAIN).iterrows():
        subset = main_units if comparison == "overall" else \
            main_units[main_units["cancer_type"] == comparison]
        tumours = subset.loc[subset["group"] == TUMOUR, "alpha"]
        lines = subset.loc[subset["group"] == CELL_LINE, "alpha"]
        recount = {"n_tumour": len(tumours), "n_cell_line": len(lines),
                   "median_delta": lines.median() - tumours.median()}
        for quantity, value in recount.items():
            records.append({"comparison": comparison, "quantity": quantity,
                            "stats_table": row[quantity], "recounted": value,
                            "agree": bool(np.isclose(row[quantity], value))})
    check = pd.DataFrame(records)
    if not check["agree"].all():
        print(check[~check["agree"]].to_string(index=False))
        raise SystemExit("The units table and the statistics table disagree; figures not drawn.")
    return check


class NumberLog:
    """Every number a figure prints, so the text can be checked against one table."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, figure: str, panel: str, item: str, quantity: str, value) -> None:
        self.rows.append({"figure": figure, "panel": panel, "item": item,
                          "quantity": quantity, "value": value})

    def save(self, path: Path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False)


# ── Small drawing helpers ──────────────────────────────────────────────────────

def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.edgecolor": "#8a897f", "axes.labelcolor": "#3d3c37",
                         "xtick.color": "#5f5e57", "ytick.color": "#5f5e57"})


def tidy(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def label_of(cancer_type: str, stacked: bool = False) -> str:
    """Display name; stacked puts a bracketed code on its own line, for narrow tick labels."""
    label = TYPE_LABELS.get(cancer_type, cancer_type)
    return label.replace(" (", "\n(") if stacked else label


def format_p(p: float) -> str:
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def box(ax, x: float, values: pd.Series, colour: str, rng: np.random.Generator,
        width: float = 0.5) -> None:
    """Box plot of one group with every unit drawn on top as a small dot.

    The box is the middle half of the units, the black line the median and the
    whiskers reach the furthest unit within 1.5 box-lengths. A group with fewer
    than MIN_BOX units gets no box, only its dots and median, because a box
    drawn from one or two values would suggest a spread that is not there.
    """
    values = values.dropna()
    if len(values) >= MIN_BOX:
        parts = ax.boxplot([values], positions=[x], widths=width, whis=1.5, patch_artist=True,
                           showfliers=False, manage_ticks=False, zorder=2)
        for patch in parts["boxes"]:
            patch.set(facecolor=to_rgba(colour, 0.28), edgecolor=colour, linewidth=1.3)
        for line in parts["whiskers"] + parts["caps"]:
            line.set(color=colour, linewidth=1.3)
        for line in parts["medians"]:
            line.set(color=INK, linewidth=2.2)
    elif len(values):
        ax.plot([x - width / 2, x + width / 2], [values.median()] * 2, color=INK, lw=2.2,
                zorder=4, solid_capstyle="butt")
    jitter = rng.uniform(-width * 0.3, width * 0.3, len(values))
    ax.scatter(x + jitter, values, s=11, color=colour, edgecolor="white", linewidth=0.4,
               alpha=0.9, zorder=3)


def normal_lines(ax, references: dict[str, tuple[float, int]]) -> None:
    for name, (median, _) in references.items():
        ax.axhline(median, color=NORMAL_LINES[name], lw=1.0, ls=(0, (4, 3)), alpha=0.75,
                   zorder=1)


def normal_handles(references: dict[str, tuple[float, int]]) -> list[Line2D]:
    return [Line2D([], [], color=NORMAL_LINES[name], lw=1.1, ls=(0, (4, 3)),
                   label=f"{name}, median (n = {n}, not tested)")
            for name, (_, n) in references.items()]


def group_handles(counts: dict[str, int] | None = None) -> list[Line2D]:
    def label(group: str) -> str:
        return group if counts is None else f"{group} (n = {counts[group]})"
    return [Patch(facecolor=to_rgba(COLOUR[g], 0.28), edgecolor=COLOUR[g], label=label(g))
            for g in (TUMOUR, CELL_LINE)]


def box_key() -> list[Line2D]:
    """How to read a box, for the legend."""
    return [Line2D([], [], color=INK, lw=2.2, label="median"),
            Line2D([], [], marker="o", ls="", color="#8a897f", markeredgecolor="white",
                   markersize=4.5, label="each dot: one patient or cell line"),
            Patch(facecolor="none", edgecolor="#8a897f",
                  label=f"box: middle half; whiskers: 1.5 \u00d7 box; under {MIN_BOX}: no box")]


def save(fig, out_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ── Figure 1: alpha by cancer type ─────────────────────────────────────────────

def figure_1(units: pd.DataFrame, references, log: NumberLog, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    tumours = units[units["group"] == TUMOUR]
    lines = units[units["group"] == CELL_LINE]
    tumour_order = tumours.groupby("cancer_type")["alpha"].median().sort_values().index.tolist()
    line_order = [t for t in tumour_order if t in set(lines["cancer_type"])] + \
        sorted(set(lines["cancer_type"]) - set(tumour_order))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [len(tumour_order), len(line_order)],
                                          "wspace": 0.06})
    for ax, data, order, group, letter in (
            (axes[0], tumours, tumour_order, TUMOUR, "A"),
            (axes[1], lines, line_order, CELL_LINE, "B")):
        normal_lines(ax, references)
        labels = []
        for x, cancer_type in enumerate(order):
            values = data.loc[data["cancer_type"] == cancer_type, "alpha"]
            box(ax, x, values, COLOUR[group], rng, width=0.55)
            labels.append(f"{label_of(cancer_type, stacked=True)}\nn = {len(values)}")
            log.add("Fig 1", letter, cancer_type, f"n {group}", len(values))
            log.add("Fig 1", letter, cancer_type, f"median alpha {group}", values.median())
        ax.set_xticks(range(len(order)), labels, fontsize=8.3)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(f"{letter}   {'Primary tumours' if group == TUMOUR else 'Cancer cell lines'}"
                     f" (n = {len(data)})", loc="left", fontsize=11.5, weight="bold", color=INK)
        tidy(ax)
        log.add("Fig 1", letter, "all", f"n {group}", len(data))
    axes[0].set_ylabel("scaling exponent α (0.1 to 10 Mb)")
    axes[1].tick_params(axis="y", length=0)
    handles = box_key() + normal_handles(references)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.1), fontsize=8.6)
    for name, (median, n) in references.items():
        log.add("Fig 1", "A, B", name, "median alpha", median)
        log.add("Fig 1", "A, B", name, "n", n)
    save(fig, out_dir, "fig1")


# ── Figure 2 and S1: matched types, and overall ────────────────────────────────

def matched_panel(ax, units: pd.DataFrame, rows: pd.DataFrame, figure: str, panel: str,
                  log: NumberLog, rng: np.random.Generator) -> None:
    """Tumour against cell line for each matched type, with delta and p above each pair."""
    types = [t for t in rows.index if t != "overall"]
    for x, cancer_type in enumerate(types):
        row = rows.loc[cancer_type]
        subset = units[units["cancer_type"] == cancer_type]
        for offset, group in ((-0.2, TUMOUR), (0.2, CELL_LINE)):
            box(ax, x + offset, subset.loc[subset["group"] == group, "alpha"], COLOUR[group],
                rng, width=0.34)
        tested = pd.notna(row["mann_whitney_p_bonferroni"])
        p_text = f"p = {format_p(row['mann_whitney_p_bonferroni'])}" if tested else "not tested"
        ax.text(x, 1.0, f"Δα = {row['median_delta']:+.3f}\n{p_text}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.2,
                color=INK if tested else MUTED, weight="bold" if tested and
                row["mann_whitney_p_bonferroni"] < 0.05 else "normal")
        for quantity in ("n_tumour", "n_cell_line", "median_delta", "mann_whitney_p_bonferroni"):
            log.add(figure, panel, cancer_type, quantity, row[quantity])
    ax.set_xticks(range(len(types)),
                  [f"{label_of(t, stacked=True)}\n{int(rows.loc[t, 'n_tumour'])} vs {int(rows.loc[t, 'n_cell_line'])}"
                   for t in types], fontsize=8.5)
    ax.set_xlim(-0.6, len(types) - 0.4)


def figure_2(units: pd.DataFrame, stats_table: pd.DataFrame, references, log: NumberLog,
             out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    rows = comparisons(stats_table, MAIN)
    main_units = units[units["group"].isin([TUMOUR, CELL_LINE])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True,
                             gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.08})

    ax = axes[0]
    normal_lines(ax, references)
    matched_panel(ax, main_units, rows, "Fig 2", "A", log, rng)
    ax.set_ylabel("scaling exponent α (0.1 to 10 Mb)")
    ax.set_title("A   Within each cancer type (tumours vs cell lines)", loc="left",
                 fontsize=11.5, weight="bold", color=INK, pad=34)
    tidy(ax)

    ax = axes[1]
    normal_lines(ax, references)
    overall = rows.loc["overall"]
    for x, group in ((0, TUMOUR), (1, CELL_LINE)):
        box(ax, x, main_units.loc[main_units["group"] == group, "alpha"], COLOUR[group], rng,
            width=0.55)
    adjusted = stats_table[(stats_table["analysis"] == MAIN)
                           & stats_table["comparison"].str.startswith("type-adjusted")].iloc[0]
    ax.text(0.5, 1.0,
            f"Δα = {overall['median_delta']:+.3f}\n"
            f"p = {format_p(overall['mann_whitney_p'])}, d = {overall['cohens_d']:.2f}",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.2,
            color=INK, weight="bold")
    ax.set_xticks([0, 1], [f"tumours\nn = {int(overall['n_tumour'])}",
                           f"cell lines\nn = {int(overall['n_cell_line'])}"], fontsize=8.5)
    ax.set_xlim(-0.6, 1.6)
    ax.tick_params(axis="y", length=0)
    ax.set_title("B   All", loc="left", fontsize=11.5, weight="bold", color=INK, pad=34)
    tidy(ax)
    for quantity in ("n_tumour", "n_cell_line", "median_delta", "mann_whitney_p", "cohens_d"):
        log.add("Fig 2", "B", "overall", quantity, overall[quantity])
    for quantity in ("adjusted_delta", "ci_low", "ci_high", "adjusted_p"):
        log.add("Fig 2", "caption", adjusted["comparison"], quantity, adjusted[quantity])

    handles = group_handles() + normal_handles(references) + box_key()
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.12), fontsize=8.5, handlelength=2.4)
    save(fig, out_dir, "fig2")


def figure_s1(units: pd.DataFrame, stats_table: pd.DataFrame, references, log: NumberLog,
              out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    rows = comparisons(stats_table, HIC_ONLY)
    hic_units = units[(units["group"] == CELL_LINE)
                      | ((units["group"] == TUMOUR) & units["hic_only"])]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    normal_lines(ax, references)
    matched_panel(ax, hic_units, rows, "Fig S1", "", log, rng)
    overall = rows.loc["overall"]
    ax.set_ylabel("scaling exponent α (0.1 to 10 Mb)")
    ax.set_title(f"Tumours measured by Hi-C only: overall Δα = "
                 f"{overall['median_delta']:+.3f}, p = {format_p(overall['mann_whitney_p'])} "
                 f"({int(overall['n_tumour'])} vs {int(overall['n_cell_line'])})",
                 loc="left", fontsize=10.5, weight="bold", color=INK, pad=34)
    for quantity in ("n_tumour", "n_cell_line", "median_delta", "mann_whitney_p"):
        log.add("Fig S1", "", "overall", quantity, overall[quantity])
    tidy(ax)
    handles = group_handles() + normal_handles(references) + box_key()
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    save(fig, out_dir, "figS1")


# ── Figures 3 and 4: contact curves ────────────────────────────────────────────

def unit_curves(curves: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """One curve per unit: each sample scaled to 1 at 100 kb, then averaged in log space."""
    log_p = np.log10(curves.pivot(index="distance_bp", columns="sample_id",
                                  values="contact_probability"))
    log_s = np.log10(log_p.index.to_numpy())
    anchor = {sample: np.interp(np.log10(ANCHOR_BP), log_s, log_p[sample]) for sample in log_p}
    scaled = log_p - pd.Series(anchor)
    return pd.DataFrame({unit: scaled[samples].mean(axis=1)
                         for unit, samples in unit_samples(units).items()})


def band(ax, curves: pd.DataFrame, colour: str, label: str, dashed: bool = False) -> None:
    """Median curve with its interquartile range shaded."""
    distance = curves.index.to_numpy()
    low, mid, high = (10 ** curves.quantile(q, axis=1) for q in (0.25, 0.5, 0.75))
    if curves.shape[1] > 1:
        ax.fill_between(distance, low, high, color=colour, alpha=0.18, lw=0)
    ax.loglog(distance, mid, color=colour, lw=2.2, ls="--" if dashed else "-", label=label)


def curve_axes(ax) -> None:
    ax.axvspan(*FIT_RANGE, color=FIT_SHADE, zorder=0)
    ax.set_xlim(2e4, 2e7)
    ax.set_xlabel("distance s (bp)")
    tidy(ax, "both")


def figure_3(units: pd.DataFrame, curves: pd.DataFrame, master: pd.DataFrame, log: NumberLog,
             out_dir: Path) -> None:
    by_group = {g: curves[units.loc[units["group"] == g, "unit"]] for g in (TUMOUR, CELL_LINE)}
    sample_class = master.set_index("sample_id")["sample_class"]
    normals = units[units["group"] == NORMAL]
    cultured = normals["samples"].map(
        lambda s: sample_class[s.split("; ")[0]] == "normal_cell_line")
    normal_curves = {"normal tissue and primary cells": curves[normals.loc[~cultured, "unit"]],
                     "cultured normal lines": curves[normals.loc[cultured, "unit"]]}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    ax = axes[0]
    for group in (TUMOUR, CELL_LINE):
        ax.loglog(by_group[group].index, 10 ** by_group[group], color=COLOUR[group], lw=0.6,
                  alpha=0.35)
    curve_axes(ax)
    ax.set_ylabel("contact probability P(s), scaled to 1 at 100 kb")
    ax.set_title("A   Every tumour and cancer cell line", loc="left", fontsize=11.5,
                 weight="bold", color=INK)
    ax.legend(handles=group_handles({g: by_group[g].shape[1] for g in by_group}),
              frameon=False, loc="lower left", fontsize=8.8)

    ax = axes[1]
    for group in (TUMOUR, CELL_LINE):
        band(ax, by_group[group], COLOUR[group], f"{group} median (n = {by_group[group].shape[1]})")
        log.add("Fig 3", "B", group, "n curves", by_group[group].shape[1])
    for name, group_curves in normal_curves.items():
        ax.loglog(group_curves.index, 10 ** group_curves.median(axis=1),
                  color=NORMAL_LINES[name], lw=1.3, ls=(0, (4, 3)),
                  label=f"{name} median (n = {group_curves.shape[1]}, not tested)")
        log.add("Fig 3", "B", name, "n curves", group_curves.shape[1])
    curve_axes(ax)
    ax.set_title("B   Group medians (shaded: middle half of units)", loc="left",
                 fontsize=11.5, weight="bold", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    save(fig, out_dir, "fig3")


def figure_4(units: pd.DataFrame, curves: pd.DataFrame, stats_table: pd.DataFrame,
             log: NumberLog, out_dir: Path) -> None:
    rows = comparisons(stats_table, MAIN)
    types = [t for t in rows.index if t != "overall"]
    columns = 4
    fig, axes = plt.subplots(int(np.ceil(len(types) / columns)), columns, figsize=(13, 6.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    axes = axes.ravel()
    for ax, cancer_type in zip(axes, types):
        row = rows.loc[cancer_type]
        for group in (TUMOUR, CELL_LINE):
            members = units.loc[(units["group"] == group)
                                & (units["cancer_type"] == cancer_type), "unit"]
            band(ax, curves[members], COLOUR[group], group)
        tested = pd.notna(row["mann_whitney_p_bonferroni"])
        p_text = f"p = {format_p(row['mann_whitney_p_bonferroni'])}" if tested else "not tested"
        ax.set_title(f"{label_of(cancer_type)}   {int(row['n_tumour'])} vs {int(row['n_cell_line'])}\n"
                     f"Δα = {row['median_delta']:+.3f}, {p_text}", loc="left",
                     fontsize=9.2, color=INK)
        curve_axes(ax)
        ax.set_xlabel("")
        for quantity in ("n_tumour", "n_cell_line", "median_delta", "mann_whitney_p_bonferroni"):
            log.add("Fig 4", cancer_type, cancer_type, quantity, row[quantity])
    for ax in axes[len(types):]:
        ax.axis("off")
    key = axes[len(types)] if len(types) < len(axes) else axes[-1]
    key.legend(handles=[Line2D([], [], color=COLOUR[g], lw=2.2, label=f"{g} median")
                        for g in (TUMOUR, CELL_LINE)], frameon=False, loc="center left",
               fontsize=9, title="shaded: middle half of units\ngrey: fitting window 0.1 to 10 Mb",
               title_fontsize=8.5)
    fig.supylabel("contact probability P(s), scaled to 1 at 100 kb", fontsize=9.5,
                  color="#3d3c37", x=0.07)
    fig.supxlabel("distance s (bp)", fontsize=9.5, color="#3d3c37", y=0.0)
    save(fig, out_dir, "fig4")


# ── Run ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    arguments = parser.parse_args()

    data_root = arguments.data_root.expanduser().resolve()
    out_dir = data_root / "figures"
    out_dir.mkdir(exist_ok=True)
    tables = load_tables(data_root)
    units, stats_table = tables["analysis_units"], tables["stats_results"]
    master = tables["master_alpha_table"]

    check = check_against_stats(units, stats_table)
    print(f"Check: {len(check)} numbers recounted from the units table, all agree with the stats.")

    style()
    log = NumberLog()
    references = normal_references(units, master)
    curves = unit_curves(tables["ps_curves"], units)
    figure_1(units, references, log, out_dir)
    figure_2(units, stats_table, references, log, out_dir)
    figure_3(units, curves, master, log, out_dir)
    figure_4(units, curves, stats_table, log, out_dir)
    figure_s1(units, stats_table, references, log, out_dir)
    log.save(out_dir / "figure_numbers.csv")
    print(f"Saved figures and figure_numbers.csv in {out_dir}")


if __name__ == "__main__":
    main()
