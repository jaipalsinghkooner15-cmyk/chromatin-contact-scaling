#!/usr/bin/env python3
"""
Build the master alpha tables: one row per sample, and one row per biological unit.

Joins the alpha values produced by the processing steps to the curated sample
annotations in data/sample_annotations.csv. That file is the single source of
truth for what each sample is: name, cancer type, sample class and whether it
is excluded. Labels written by the processing steps are kept only for
comparison and are never used on their own.

Inputs (all under --data-root, see RESULTS_FILES for details)
    cellline_alpha_results.csv                   02_process_cellline_hic.py
    gbm_patient_alpha_results.csv                GBM_processing.py (GSE229962)
    gbm_cellline_alpha_results.csv               03_process_gbm_celllines.py (GSE280831)
    TCGA_HiChIP_hic/TCGA_powerlaw_results.csv    TCGA HiChIP processing
Plus data/sample_annotations.csv, kept in this repository.

Outputs
    <data-root>/master_alpha_table.csv     one row per sample, excluded ones included
    <data-root>/master_alpha_by_unit.csv   one row per biological unit, see below

A biological unit is one cell line, culture or tumour. Replicates of the same
cells (e.g. A549 rep1 to rep4) share a unit and their alpha values are averaged.
Excluded samples are left out of the unit table. parent_line records shared
origin (e.g. a relapse culture and its primary), so related units can be
grouped further at the statistics stage without editing this table.

Samples that have no annotation yet keep their pipeline label and are marked
reviewed = False, so the clean-up still to do is always visible in the report.

Sign convention
    The processing steps store alpha as a positive number (alpha = -slope).
    The master table stores the slope itself, so values are negative and a
    more negative value means steeper decay, as reported in the paper.

Usage
    python preprocess/04_build_master_alpha_table.py
    python preprocess/04_build_master_alpha_table.py --data-root "C:/path/to/Dissertation"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "sample_annotations.csv"
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"
OUTPUT_NAME = "master_alpha_table.csv"
UNIT_OUTPUT_NAME = "master_alpha_by_unit.csv"

SAMPLE_CLASSES = {
    "cancer_cell_line",
    "normal_cell_line",
    "normal_primary",
    "tumour",
    "mixture",
    "unverified",
}

ANNOTATION_COLUMNS = [
    "sample_id", "dataset", "gsm", "sample_name", "biological_unit",
    "parent_line", "sample_class", "cancer_type", "tissue", "assay",
    "exclude", "exclusion_reason", "evidence", "notes",
]

# Every results file is converted to these columns before anything else happens.
STANDARD_RESULT_COLUMNS = [
    "sample_id", "dataset", "pipeline_cancer_type", "genome",
    "resolution_bp", "alpha", "r_squared", "n_points",
]


@dataclass(frozen=True)
class ResultsFile:
    """How to read one results file written by a processing step.

    columns  maps the file's own column names to the standard names above.
    fixed    supplies standard columns the file does not record. Each value
             has a comment saying where it comes from; None means unknown.
    sample_suffix is trimmed from the end of every sample name, if present.
    """

    path: str
    producer: str
    pipeline_sample_class: str
    columns: dict[str, str]
    fixed: dict[str, object] = field(default_factory=dict)
    sample_suffix: str = ""


RESULTS_FILES = [
    ResultsFile(
        path="cellline_alpha_results.csv",
        producer="02_process_cellline_hic.py",
        pipeline_sample_class="cancer_cell_line",
        columns={
            "sample": "sample_id",
            "source": "dataset",
            "cancer_type": "pipeline_cancer_type",
            "genome": "genome",
            "resolution": "resolution_bp",
            "alpha": "alpha",
            "r_squared": "r_squared",
            "n_points": "n_points",
        },
    ),
    ResultsFile(
        path="gbm_patient_alpha_results.csv",
        producer="GBM_processing.py",
        pipeline_sample_class="tumour",
        columns={
            "sample": "sample_id",
            "source": "dataset",
            "cancer_type": "pipeline_cancer_type",
            "alpha": "alpha",
            "r_squared": "r_squared",
        },
        fixed={
            "genome": "hg38",          # Xie et al. 2024, Methods: aligned to GRCh38
            "resolution_bp": 50_000,   # RESOLUTION in GBM_processing.py
            "n_points": None,          # not recorded by GBM_processing.py
        },
    ),
    ResultsFile(
        path="gbm_cellline_alpha_results.csv",
        producer="03_process_gbm_celllines.py",
        pipeline_sample_class="cancer_cell_line",
        columns={
            "sample": "sample_id",
            "source": "dataset",
            "cancer_type": "pipeline_cancer_type",
            "alpha": "alpha",
            "r_squared": "r_squared",
        },
        fixed={
            "genome": "hg38",          # stated in 03_process_gbm_celllines.py
            "resolution_bp": 50_000,   # RESOLUTION in 03_process_gbm_celllines.py
            "n_points": None,          # not recorded by 03_process_gbm_celllines.py
        },
    ),
    ResultsFile(
        path="TCGA_HiChIP_hic/TCGA_powerlaw_results.csv",
        producer="the TCGA HiChIP processing script",
        pipeline_sample_class="tumour",
        columns={
            "Sample": "sample_id",
            "Cancer_Type": "pipeline_cancer_type",
            "Alpha": "alpha",
            "R_squared": "r_squared",
            "n_points": "n_points",
        },
        fixed={
            "dataset": "TCGA",
            "genome": None,            # processing script not in the repository yet
            "resolution_bp": None,     # processing script not in the repository yet
        },
        sample_suffix="_H3K27ac.allValidPairs.hic",
    ),
]

OUTPUT_COLUMNS = [
    "sample_id", "dataset", "gsm", "sample_name", "biological_unit",
    "parent_line", "sample_class", "cancer_type", "tissue", "assay",
    "genome", "resolution_bp", "alpha", "r_squared", "n_points",
    "exclude", "exclusion_reason", "reviewed", "pipeline_cancer_type", "pipeline_sample_class",
    "evidence", "notes",
]

TEXT_COLUMNS = [
    "gsm", "sample_name", "biological_unit", "parent_line", "sample_class",
    "cancer_type", "tissue", "assay", "exclusion_reason", "evidence", "notes",
]


def require_columns(table: pd.DataFrame, expected: list[str], source: str) -> None:
    """Stop with a clear message if an input file is missing expected columns."""
    missing = [column for column in expected if column not in table.columns]
    if missing:
        raise SystemExit(f"{source} is missing columns: {', '.join(missing)}")


def load_annotations(path: Path) -> pd.DataFrame:
    """Read the curated annotations and check every row is well formed."""
    if not path.exists():
        raise SystemExit(f"Annotation file not found: {path}")

    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(table, ANNOTATION_COLUMNS, path.name)
    table = table.apply(lambda column: column.str.strip())

    problems = []

    duplicated = table.loc[table["sample_id"].duplicated(), "sample_id"]
    if not duplicated.empty:
        problems.append(f"duplicate sample_id: {', '.join(duplicated)}")

    unknown_classes = sorted(set(table["sample_class"]) - SAMPLE_CLASSES)
    if unknown_classes:
        problems.append(f"unknown sample_class: {', '.join(unknown_classes)}")

    bad_exclude = sorted(set(table["exclude"]) - {"yes", "no"})
    if bad_exclude:
        problems.append(f"exclude must be yes or no, found: {', '.join(bad_exclude)}")

    no_reason = table.loc[
        (table["exclude"] == "yes") & (table["exclusion_reason"] == ""), "sample_id"
    ]
    if not no_reason.empty:
        problems.append(f"excluded without a reason: {', '.join(no_reason)}")

    if problems:
        raise SystemExit(f"Problems in {path.name}:\n  - " + "\n  - ".join(problems))

    table["exclude"] = table["exclude"] == "yes"
    return table


def load_results(data_root: Path, spec: ResultsFile) -> pd.DataFrame:
    """Read one results file, check it, and convert it to the standard columns."""
    path = data_root / spec.path
    if not path.exists():
        raise SystemExit(
            f"Not found: {path}\n"
            f"Run {spec.producer} first, or point --data-root at the right folder."
        )

    raw = pd.read_csv(path)
    require_columns(raw, list(spec.columns), path.name)

    results = raw[list(spec.columns)].rename(columns=spec.columns)
    for column, value in spec.fixed.items():
        results[column] = value
    require_columns(results, STANDARD_RESULT_COLUMNS, f"Settings for {spec.path}")

    results["sample_id"] = results["sample_id"].astype(str).str.strip()
    if spec.sample_suffix:
        results["sample_id"] = results["sample_id"].str.removesuffix(spec.sample_suffix)

    if (results["alpha"] <= 0).any():
        raise SystemExit(
            f"{path.name}: expected positive alpha values (alpha = -slope). "
            "The file may already use the negative convention; check it before continuing."
        )

    duplicated = results.loc[results["sample_id"].duplicated(), "sample_id"]
    if not duplicated.empty:
        raise SystemExit(f"{path.name}: duplicate samples: {', '.join(duplicated)}")

    results["dataset"] = results["dataset"].astype(str).str.strip()
    results["pipeline_cancer_type"] = results["pipeline_cancer_type"].astype(str).str.strip()
    results["alpha"] = -results["alpha"]
    results["resolution_bp"] = results["resolution_bp"].astype("Int64")
    results["n_points"] = results["n_points"].astype("Int64")
    results["pipeline_sample_class"] = spec.pipeline_sample_class

    return results[STANDARD_RESULT_COLUMNS + ["pipeline_sample_class"]]


def combine_results(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack the results files and make sure no sample appears in two of them."""
    combined = pd.concat(tables, ignore_index=True)
    duplicated = combined.loc[combined["sample_id"].duplicated(), "sample_id"]
    if not duplicated.empty:
        raise SystemExit(f"Samples found in more than one results file: {', '.join(duplicated)}")
    return combined


def apply_annotations(alphas: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Attach annotations to each sample; unannotated samples keep pipeline labels."""
    annotations = annotations.rename(columns={"dataset": "annotated_dataset"})
    table = alphas.merge(annotations, on="sample_id", how="left", indicator=True)

    reviewed = table["_merge"] == "both"
    wrong_dataset = reviewed & (table["dataset"] != table["annotated_dataset"])
    if wrong_dataset.any():
        clashes = table.loc[wrong_dataset, ["sample_id", "dataset", "annotated_dataset"]]
        raise SystemExit(
            "Annotation dataset does not match the results file:\n"
            + clashes.to_string(index=False)
        )

    table["reviewed"] = reviewed
    table[TEXT_COLUMNS] = table[TEXT_COLUMNS].fillna("")
    table.loc[~reviewed, "sample_name"] = table.loc[~reviewed, "sample_id"]
    table.loc[~reviewed, "biological_unit"] = table.loc[~reviewed, "sample_id"]
    table.loc[~reviewed, "cancer_type"] = table.loc[~reviewed, "pipeline_cancer_type"]
    table.loc[~reviewed, "sample_class"] = table.loc[~reviewed, "pipeline_sample_class"]
    table["exclude"] = table["exclude"].eq(True)

    return table[OUTPUT_COLUMNS]


def build_unit_table(table: pd.DataFrame) -> pd.DataFrame:
    """Average replicates so that each biological unit appears once."""
    included = table[~table["exclude"]]

    no_unit = included.loc[included["biological_unit"] == "", "sample_id"]
    if not no_unit.empty:
        raise SystemExit(f"Included samples without a biological_unit: {', '.join(no_unit)}")

    # Every sample in a unit must agree on what the unit is.
    shared = ["dataset", "parent_line", "sample_class", "cancer_type", "tissue"]
    disagreements = included.groupby("biological_unit")[shared].nunique()
    conflicting = disagreements[(disagreements > 1).any(axis=1)]
    if not conflicting.empty:
        raise SystemExit(
            "Samples in the same biological_unit disagree on:\n" + conflicting.to_string()
        )

    units = included.groupby("biological_unit").agg(
        dataset=("dataset", "first"),
        parent_line=("parent_line", "first"),
        sample_class=("sample_class", "first"),
        cancer_type=("cancer_type", "first"),
        tissue=("tissue", "first"),
        alpha=("alpha", "mean"),
        alpha_sd=("alpha", "std"),
        n_samples=("sample_id", "size"),
        sample_ids=("sample_id", lambda ids: "; ".join(sorted(ids))),
    )
    return units.reset_index()


def print_section(title: str, rows: pd.DataFrame) -> None:
    print(f"\n{title} ({len(rows)})")
    print("-" * (len(title) + len(str(len(rows))) + 3))
    print(rows.to_string(index=False) if not rows.empty else "none")


def print_report(table: pd.DataFrame, annotations: pd.DataFrame) -> None:
    """Show what changed, what is excluded and what still needs review."""
    reviewed = table[table["reviewed"]]

    # A name that is only a shortened form of the ID (e.g. TCGA) is not a rename.
    renamed = pd.Series(
        [not sample_id.startswith(name)
         for sample_id, name in zip(reviewed["sample_id"], reviewed["sample_name"], strict=True)],
        index=reviewed.index,
        dtype=bool,
    )
    changed = (
        (reviewed["cancer_type"] != reviewed["pipeline_cancer_type"])
        | (reviewed["sample_class"] != reviewed["pipeline_sample_class"])
        | renamed
    )
    relabel_columns = ["sample_id", "sample_name", "pipeline_cancer_type", "cancer_type",
                       "pipeline_sample_class", "sample_class"]
    print_section(
        "Relabelled samples",
        reviewed.loc[changed, relabel_columns]
        .replace({"cancer_type": {"": "(none)"}})
        .rename(columns={
            "pipeline_cancer_type": "old_type",
            "cancer_type": "new_type",
            "pipeline_sample_class": "old_class",
            "sample_class": "new_class",
        }),
    )

    print_section(
        "Excluded samples",
        table.loc[table["exclude"], ["sample_id", "sample_name", "exclusion_reason"]],
    )

    unmatched = annotations.loc[~annotations["sample_id"].isin(table["sample_id"]), "sample_id"]
    if not unmatched.empty:
        print(f"\nWARNING: annotated but not in any results file: {', '.join(unmatched)}")

    unreviewed = table[~table["reviewed"]]
    print_section(
        "Still to review, by dataset",
        unreviewed.groupby("dataset").size().rename("samples").reset_index(),
    )

    included = table[~table["exclude"]]
    counts = pd.crosstab(
        included["cancer_type"].replace("", "(none)"), included["sample_class"]
    )
    print("\nIncluded samples by cancer type and class (before replicate merging)")
    print("-------------------------------------------------------------------")
    print(counts.to_string())

    print(f"\nTotal samples: {len(table)} | reviewed: {len(reviewed)} | "
          f"excluded: {int(table['exclude'].sum())} | to review: {len(unreviewed)}")


def print_unit_report(units: pd.DataFrame) -> None:
    """Show which samples were merged and which units share an origin."""
    print_section(
        "Replicates merged into one unit",
        units.loc[units["n_samples"] > 1, ["biological_unit", "n_samples", "sample_ids"]],
    )

    # Distinct units with a common parent: the grouping decision for the statistics stage.
    with_parent = units[units["parent_line"] != ""]
    shared = with_parent.groupby("parent_line")["biological_unit"].agg(["size", list])
    shared = shared[shared["size"] > 1].reset_index()
    shared["units"] = shared["list"].map(", ".join)
    print_section(
        "Distinct units that share an origin (kept separate for now)",
        shared[["parent_line", "units"]],
    )

    counts = pd.crosstab(units["cancer_type"].replace("", "(none)"), units["sample_class"])
    print("\nBiological units by cancer type and class (after replicate merging)")
    print("------------------------------------------------------------------")
    print(counts.to_string())
    print(f"\nTotal units: {len(units)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"folder holding the results CSVs (default: {DEFAULT_DATA_ROOT})",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    data_root = arguments.data_root.expanduser().resolve()

    annotations = load_annotations(ANNOTATIONS_PATH)
    alphas = combine_results([load_results(data_root, spec) for spec in RESULTS_FILES])
    table = apply_annotations(alphas, annotations)

    units = build_unit_table(table)

    print_report(table, annotations)
    print_unit_report(units)

    for name, frame in [(OUTPUT_NAME, table), (UNIT_OUTPUT_NAME, units)]:
        output_path = data_root / name
        frame.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
