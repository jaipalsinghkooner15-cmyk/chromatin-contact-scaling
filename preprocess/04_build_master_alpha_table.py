#!/usr/bin/env python3
"""
Build the per-sample master alpha table.

Joins the alpha values produced by the processing steps to the curated sample
annotations in data/sample_annotations.csv. That file is the single source of
truth for what each sample is: cell line, cancer type, sample class and
whether it is excluded. Labels written by the processing steps are kept only
for comparison and are never used on their own.

Inputs
    <data-root>/cellline_alpha_results.csv     written by 02_process_cellline_hic.py
    <data-root>/gbm_patient_alpha_results.csv  GSE229962 GSCs, from GBM_processing.py
    <data-root>/gbm_cellline_alpha_results.csv GSE280831 GSCs, from 03_process_gbm_celllines.py
    data/sample_annotations.csv                curated, kept in this repository

Output
    <data-root>/master_alpha_table.csv

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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_PATH = REPO_ROOT / "data" / "sample_annotations.csv"
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"

CELLLINE_RESULTS_NAME = "cellline_alpha_results.csv"
OUTPUT_NAME = "master_alpha_table.csv"

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

CELLLINE_RESULT_COLUMNS = [
    "cancer_type", "sample", "source", "genome", "resolution",
    "alpha", "r_squared", "n_points",
]

SHORT_RESULT_COLUMNS = ["sample", "cancer_type", "source", "alpha", "r_squared"]


@dataclass(frozen=True)
class ShortResultsFile:
    """A results file with only the five SHORT_RESULT_COLUMNS.

    These files do not record genome or resolution, so both are fixed here,
    each with a note on where the value comes from.
    """

    name: str
    producer: str
    genome: str
    resolution_bp: int
    pipeline_sample_class: str


SHORT_RESULTS_FILES = [
    ShortResultsFile(
        name="gbm_patient_alpha_results.csv",
        producer="GBM_processing.py",
        genome="hg38",           # Xie et al. 2024, Methods: aligned to GRCh38
        resolution_bp=50_000,    # RESOLUTION in GBM_processing.py
        pipeline_sample_class="tumour",
    ),
    ShortResultsFile(
        name="gbm_cellline_alpha_results.csv",
        producer="03_process_gbm_celllines.py",
        genome="hg38",           # stated in 03_process_gbm_celllines.py
        resolution_bp=50_000,    # RESOLUTION in 03_process_gbm_celllines.py
        pipeline_sample_class="cancer_cell_line",
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


def read_results_file(path: Path, expected: list[str], producer: str) -> pd.DataFrame:
    """Read one results file and check it before any values are used."""
    if not path.exists():
        raise SystemExit(
            f"Not found: {path}\n"
            f"Run {producer} first, or point --data-root at the right folder."
        )

    results = pd.read_csv(path)
    require_columns(results, expected, path.name)

    if (results["alpha"] <= 0).any():
        raise SystemExit(
            f"{path.name}: expected positive alpha values (alpha = -slope). "
            "The file may already use the negative convention; check it before continuing."
        )

    duplicated = results.loc[results["sample"].duplicated(), "sample"]
    if not duplicated.empty:
        raise SystemExit(f"{path.name}: duplicate samples: {', '.join(duplicated)}")

    return results


def load_cellline_alphas(data_root: Path) -> pd.DataFrame:
    """Cell line results from 02_process_cellline_hic.py, alpha made negative."""
    results = read_results_file(
        data_root / CELLLINE_RESULTS_NAME, CELLLINE_RESULT_COLUMNS, "02_process_cellline_hic.py"
    )
    return pd.DataFrame({
        "sample_id": results["sample"].astype(str).str.strip(),
        "dataset": results["source"].astype(str).str.strip(),
        "pipeline_cancer_type": results["cancer_type"].astype(str).str.strip(),
        # The pipeline treated everything in this file as a cancer cell line.
        "pipeline_sample_class": "cancer_cell_line",
        "genome": results["genome"],
        "resolution_bp": results["resolution"].astype(int),
        "alpha": -results["alpha"],
        "r_squared": results["r_squared"],
        "n_points": results["n_points"].astype("Int64"),
    })


def load_short_alphas(data_root: Path, spec: ShortResultsFile) -> pd.DataFrame:
    """Results from one short-format file, alpha made negative."""
    results = read_results_file(data_root / spec.name, SHORT_RESULT_COLUMNS, spec.producer)
    return pd.DataFrame({
        "sample_id": results["sample"].astype(str).str.strip(),
        "dataset": results["source"].astype(str).str.strip(),
        "pipeline_cancer_type": results["cancer_type"].astype(str).str.strip(),
        "pipeline_sample_class": spec.pipeline_sample_class,
        "genome": spec.genome,
        "resolution_bp": spec.resolution_bp,
        "alpha": -results["alpha"],
        "r_squared": results["r_squared"],
        # Not recorded in short-format files.
        "n_points": pd.array([pd.NA] * len(results), dtype="Int64"),
    })


def combine_results(*tables: pd.DataFrame) -> pd.DataFrame:
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
    table.loc[~reviewed, "cancer_type"] = table.loc[~reviewed, "pipeline_cancer_type"]
    table.loc[~reviewed, "sample_class"] = table.loc[~reviewed, "pipeline_sample_class"]
    table["exclude"] = table["exclude"].eq(True)

    return table[OUTPUT_COLUMNS]


def print_section(title: str, rows: pd.DataFrame) -> None:
    print(f"\n{title} ({len(rows)})")
    print("-" * (len(title) + len(str(len(rows))) + 3))
    print(rows.to_string(index=False) if not rows.empty else "none")


def print_report(table: pd.DataFrame, annotations: pd.DataFrame) -> None:
    """Show what changed, what is excluded and what still needs review."""
    reviewed = table[table["reviewed"]]

    changed = (
        (reviewed["cancer_type"] != reviewed["pipeline_cancer_type"])
        | (reviewed["sample_class"] != reviewed["pipeline_sample_class"])
        | (reviewed["sample_name"] != reviewed["sample_id"])
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
    alphas = combine_results(
        load_cellline_alphas(data_root),
        *(load_short_alphas(data_root, spec) for spec in SHORT_RESULTS_FILES),
    )
    table = apply_annotations(alphas, annotations)

    print_report(table, annotations)

    output_path = data_root / OUTPUT_NAME
    table.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
