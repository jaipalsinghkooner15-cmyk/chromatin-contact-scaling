#!/usr/bin/env python3
"""
Compute the contact scaling exponent alpha for every sample with one method.

Every sample, tumour or cell line, whatever its file format, goes through the
same steps:

    1. Raw contact counts at 10 kb, autosomes only, same chromosome only.
       No balancing or other normalisation is applied to any sample.
    2. For each distance d (in 10 kb steps), add up the contacts at that
       distance and divide by the number of bin pairs at that distance:
           P(d) = total contacts at distance d / number of bin pairs at distance d
       Bin pairs with no contacts count in the denominator, so zeros are included.
       The first two diagonals (0 and 10 kb) are skipped.
    3. Group distances into 200 log-spaced bins with fixed edges from 20 kb to
       20 Mb (the same edges for every sample).
    4. Fit log10 P(s) against log10 s by least squares between 0.1 and 10 Mb.
       alpha = -slope, stored as a positive number like the older results files.

Inputs
    data/input_files.csv   one row per sample: sample_id, dataset, file, format, genome
    --search folders       where the input files are looked up by name

Outputs (under --data-root)
    scaling_cache/<sample_id>.csv   contacts and bin pairs per distance, one file per sample
    alpha_results.csv               one row per sample: alpha, R squared, points, depth
    ps_curves.csv                   log-binned P(s) curve for every sample, for figures

A sample whose cache file exists is not recomputed unless --force is given, so
an interrupted run can simply be started again.

Formats in data/input_files.csv
    hic, mcool, cool       read with hictkpy at 10 kb
    hic_per_chromosome     one .hic per chromosome; file contains {chrom}, e.g. BxPC3.{chrom}.hic
    pairs_4dn              4DN .pairs(.gz); columns and chromosome sizes read from its header
    validpairs_hicpro      HiC-Pro allValidPairs; chromosome sizes from GENOME_SIZES
    matrix_hicpro          HiC-Pro sparse matrix (bin, bin, count) at 10 kb with genome-wide
                           bin numbers; the bin layout is rebuilt from GENOME_SIZES and checked

Usage
    python preprocess/02_compute_alpha.py --check --search FOLDER --search FOLDER
    python preprocess/02_compute_alpha.py --search FOLDER --search FOLDER
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_PATH = REPO_ROOT / "data" / "input_files.csv"
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"

RESOLUTION = 10_000
MIN_DIAGONAL = 2                                  # skip distances 0 and 10 kb
MAX_DISTANCE = 20_000_000
MAX_DIAGONAL = MAX_DISTANCE // RESOLUTION
FIT_MIN, FIT_MAX = 100_000, 10_000_000
N_LOG_BINS = 200
LOG_BIN_EDGES = np.logspace(
    np.log10(MIN_DIAGONAL * RESOLUTION), np.log10(MAX_DISTANCE), N_LOG_BINS + 1
)
AUTOSOMES = [str(n) for n in range(1, 23)]
PAIRS_CHUNK_ROWS = 5_000_000
LAYOUT_CHECK_BAND = 10                            # bins either side of the diagonal

# Chromosome sizes (UCSC), in the order HiC-Pro numbers its genome-wide bins.
GENOME_SIZES = {
    "hg19": {
        "chr1": 249250621, "chr2": 243199373, "chr3": 198022430, "chr4": 191154276,
        "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
        "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
        "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
        "chr21": 48129895, "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
    },
    "hg38": {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
        "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
        "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
        "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
        "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    },
}

FORMATS = {"hic", "mcool", "cool", "hic_per_chromosome", "pairs_4dn",
           "validpairs_hicpro", "matrix_hicpro"}
INPUT_COLUMNS = ["sample_id", "dataset", "file", "format", "genome"]


# ── Counting contacts by distance ──────────────────────────────────────────────

@dataclass
class DistanceTotals:
    """Contacts and bin pairs at each distance (in bins), summed over autosomes."""

    contacts: np.ndarray = field(default_factory=lambda: np.zeros(MAX_DIAGONAL + 1))
    bin_pairs: np.ndarray = field(default_factory=lambda: np.zeros(MAX_DIAGONAL + 1))

    def add_chromosome(self, n_bins: int) -> None:
        """Count every bin pair of one chromosome, with or without contacts."""
        diagonals = np.arange(MAX_DIAGONAL + 1)
        self.bin_pairs += np.clip(n_bins - diagonals, 0, None)

    def add_contacts(self, diagonals: np.ndarray, counts: np.ndarray) -> None:
        keep = (diagonals >= 0) & (diagonals <= MAX_DIAGONAL)
        self.contacts += np.bincount(
            diagonals[keep].astype(np.int64), weights=counts[keep], minlength=MAX_DIAGONAL + 1
        )


def autosome(chrom: str) -> str | None:
    """'chr7' or '7' gives '7'; anything that is not an autosome gives None."""
    name = chrom.removeprefix("chr")
    return name if name in AUTOSOMES else None


def n_bins(size: int) -> int:
    return -(-size // RESOLUTION)


# ── Readers, one per format ────────────────────────────────────────────────────

def read_hictk_file(path: Path, totals: DistanceTotals, only_chrom: str | None = None) -> None:
    """Raw counts from a .hic, .mcool or .cool file at RESOLUTION."""
    import hictkpy

    handle = hictkpy.File(str(path), RESOLUTION)
    for chrom, size in handle.chromosomes().items():
        name = autosome(chrom)
        if name is None or (only_chrom is not None and name != only_chrom):
            continue
        totals.add_chromosome(n_bins(size))
        matrix = handle.fetch(chrom, diagonal_band_width=MAX_DIAGONAL + 1).to_coo()
        totals.add_contacts(matrix.col - matrix.row, matrix.data.astype(np.float64))


def read_hic_per_chromosome(files: dict[str, Path], totals: DistanceTotals) -> None:
    """One .hic file per chromosome; only that chromosome is read from each."""
    for name in AUTOSOMES:
        read_hictk_file(files[name], totals, only_chrom=name)


def open_text(path: Path):
    return gzip.open(path, "rt") if path.name.endswith(".gz") else open(path)


def read_4dn_header(path: Path) -> tuple[list[str], dict[str, int], int]:
    """Column names, chromosome sizes and number of header lines of a 4DN pairs file."""
    columns, sizes, header_lines = [], {}, 0
    with open_text(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            header_lines += 1
            if line.startswith("#columns:"):
                columns = line.split()[1:]
            elif line.startswith("#chromsize:"):
                _, chrom, size = line.split()
                sizes[chrom] = int(size)
    if not {"chrom1", "pos1", "chrom2", "pos2"} <= set(columns):
        raise ValueError(f"{path.name}: header has no chrom1/pos1/chrom2/pos2 columns")
    return columns, sizes, header_lines


def read_pairs(path: Path, positions: list[int], sizes: dict[str, int],
               totals: DistanceTotals, header_lines: int = 0) -> None:
    """Contact pairs (chrom1, pos1, chrom2, pos2 at the given column positions)."""
    autosome_sizes = {c: s for c, s in sizes.items() if autosome(c)}
    if len(autosome_sizes) != len(AUTOSOMES):
        raise ValueError(f"{path.name}: expected 22 autosome sizes, found {len(autosome_sizes)}")
    for size in autosome_sizes.values():
        totals.add_chromosome(n_bins(size))

    chunks = pd.read_csv(
        path, sep="\t", header=None, skiprows=header_lines, usecols=positions,
        chunksize=PAIRS_CHUNK_ROWS, dtype={positions[0]: str, positions[2]: str},
    )
    for chunk in chunks:
        chrom1, pos1, chrom2, pos2 = (chunk[p] for p in positions)
        cis = (chrom1 == chrom2) & chrom1.isin(autosome_sizes.keys())
        if not cis.any():
            continue
        chrom, pos1, pos2 = chrom1[cis], pos1[cis].to_numpy(), pos2[cis].to_numpy()
        limits = chrom.map(autosome_sizes).to_numpy()
        if (np.maximum(pos1, pos2) > limits).any():
            raise ValueError(f"{path.name}: positions beyond chromosome ends, wrong genome?")
        diagonals = np.abs(pos2 // RESOLUTION - pos1 // RESOLUTION)
        totals.add_contacts(diagonals, np.ones(len(diagonals)))


def read_hicpro_matrix(path: Path, genome: str, totals: DistanceTotals) -> None:
    """HiC-Pro sparse matrix with 1-based genome-wide bin numbers.

    The bin layout is rebuilt from GENOME_SIZES. If it is wrong, even by one
    bin, the dense band along the diagonal runs across the assumed chromosome
    boundaries. The reader counts pixels within LAYOUT_CHECK_BAND bins of the
    diagonal that straddle a boundary. With the right layout these can only be
    rare trans contacts (well under one per boundary in real data), so more
    than two per chromosome with data stops the run.

    Bins after the last chromosome in GENOME_SIZES (HiC-Pro often appends
    chrM) are read as one extra, non-autosomal chromosome.
    """
    sizes = GENOME_SIZES[genome]
    bins_per_chrom = np.array([n_bins(s) for s in sizes.values()])
    starts = np.concatenate([[0], np.cumsum(bins_per_chrom)])
    for name, count in zip((autosome(c) for c in sizes), bins_per_chrom, strict=True):
        if name is not None:
            totals.add_chromosome(int(count))
    # One entry per chromosome in the layout, plus one for anything after it.
    is_autosome = np.array([autosome(c) is not None for c in sizes] + [False])

    straddling = 0
    chromosomes_with_data: set[int] = set()
    chunks = pd.read_csv(path, sep=r"\s+", header=None, names=["i", "j", "count"],
                         chunksize=PAIRS_CHUNK_ROWS)
    for chunk in chunks:
        i = chunk["i"].to_numpy() - 1
        j = chunk["j"].to_numpy() - 1
        chrom_i = np.searchsorted(starts, i, side="right") - 1
        chrom_j = np.searchsorted(starts, j, side="right") - 1
        near = np.abs(j - i) <= LAYOUT_CHECK_BAND
        straddling += int((near & (chrom_i != chrom_j)).sum())
        cis = (chrom_i == chrom_j) & is_autosome[chrom_i]
        chromosomes_with_data.update(np.unique(chrom_i[chrom_i == chrom_j]).tolist())
        totals.add_contacts(np.abs(j[cis] - i[cis]), chunk["count"].to_numpy()[cis].astype(float))

    if straddling > 2 * len(chromosomes_with_data):
        raise ValueError(
            f"{path.name}: {straddling} near-diagonal contacts cross chromosome boundaries; "
            f"the {genome} bin layout does not match this file"
        )


def compute_totals(row: pd.Series, files: list[Path]) -> DistanceTotals:
    totals = DistanceTotals()
    file_format, genome = row["format"], row["genome"]
    if file_format in ("hic", "mcool", "cool"):
        read_hictk_file(files[0], totals)
    elif file_format == "hic_per_chromosome":
        read_hic_per_chromosome(dict(zip(AUTOSOMES, files, strict=True)), totals)
    elif file_format == "pairs_4dn":
        columns, sizes, header_lines = read_4dn_header(files[0])
        positions = [columns.index(c) for c in ("chrom1", "pos1", "chrom2", "pos2")]
        read_pairs(files[0], positions, sizes, totals, header_lines)
    elif file_format == "validpairs_hicpro":
        read_pairs(files[0], [1, 2, 4, 5], GENOME_SIZES[genome], totals)
    elif file_format == "matrix_hicpro":
        read_hicpro_matrix(files[0], genome, totals)
    return totals


# ── From counts to alpha ───────────────────────────────────────────────────────

def scaling_curve(
    diagonals: np.ndarray, contacts: np.ndarray, bin_pairs: np.ndarray
) -> pd.DataFrame:
    """Log-binned P(s): contacts / bin pairs within each fixed log bin."""
    per_distance = pd.DataFrame({
        "distance": diagonals * RESOLUTION, "contacts": contacts, "bin_pairs": bin_pairs,
    })
    per_distance = per_distance[
        (per_distance["distance"] >= MIN_DIAGONAL * RESOLUTION) & (per_distance["bin_pairs"] > 0)
    ]
    per_distance["log_bin"] = np.clip(
        np.digitize(per_distance["distance"], LOG_BIN_EDGES) - 1, 0, N_LOG_BINS - 1
    )
    per_distance["log_distance"] = np.log10(per_distance["distance"])
    curve = per_distance.groupby("log_bin").agg(
        log_distance=("log_distance", "mean"),
        contacts=("contacts", "sum"),
        bin_pairs=("bin_pairs", "sum"),
    )
    curve["distance_bp"] = 10 ** curve["log_distance"]
    curve["contact_probability"] = curve["contacts"] / curve["bin_pairs"]
    columns = ["distance_bp", "contacts", "bin_pairs", "contact_probability"]
    return curve[columns].reset_index(drop=True)


def fit_alpha(curve: pd.DataFrame) -> dict:
    """Least-squares fit of log10 P(s) on log10 s between FIT_MIN and FIT_MAX."""
    in_range = curve[(curve["distance_bp"] >= FIT_MIN) & (curve["distance_bp"] <= FIT_MAX)]
    empty = int((in_range["contacts"] == 0).sum())
    usable = in_range[in_range["contacts"] > 0]
    if len(usable) < 10:
        raise ValueError(f"only {len(usable)} usable points in the fit range")
    fit = stats.linregress(np.log10(usable["distance_bp"]), np.log10(usable["contact_probability"]))
    return {
        "alpha": -fit.slope,
        "r_squared": fit.rvalue ** 2,
        "n_points": len(usable),
        "empty_fit_bins": empty,
    }


# ── Files, cache and run ───────────────────────────────────────────────────────

def index_files(folders: list[Path]) -> dict[str, list[Path]]:
    """Every file name under the search folders, with all the places it appears."""
    index: dict[str, list[Path]] = {}
    for folder in folders:
        for directory, _, names in os.walk(folder):
            for name in names:
                index.setdefault(name, []).append(Path(directory) / name)
    return index


def locate(name: str, index: dict[str, list[Path]]) -> Path:
    """The single file with this name; identical copies of it are allowed."""
    found = index.get(name, [])
    if not found:
        raise FileNotFoundError(f"not found in the search folders: {name}")
    if len({p.stat().st_size for p in found}) > 1:
        places = "\n    ".join(str(p) for p in found)
        raise FileNotFoundError(f"{name} exists in different versions:\n    {places}")
    return found[0]


def locate_inputs(row: pd.Series, index: dict[str, list[Path]]) -> list[Path]:
    if row["format"] == "hic_per_chromosome":
        return [locate(row["file"].format(chrom=f"chr{n}"), index) for n in AUTOSOMES]
    return [locate(row["file"], index)]


def load_inputs(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in INPUT_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {', '.join(missing)}")
    problems = []
    duplicated = table.loc[table["sample_id"].duplicated(), "sample_id"]
    if not duplicated.empty:
        problems.append("duplicate sample_id: " + ", ".join(duplicated))
    bad_format = sorted(set(table["format"]) - FORMATS)
    if bad_format:
        problems.append("unknown format: " + ", ".join(bad_format))
    needs_genome = table["format"].isin(["validpairs_hicpro", "matrix_hicpro"])
    bad_genome = table.loc[needs_genome & ~table["genome"].isin(GENOME_SIZES), "sample_id"]
    if not bad_genome.empty:
        problems.append("genome must be hg19 or hg38 for: " + ", ".join(bad_genome))
    if problems:
        raise SystemExit(f"Problems in {path.name}:\n  - " + "\n  - ".join(problems))
    return table


def cache_path(data_root: Path, sample_id: str) -> Path:
    return data_root / "scaling_cache" / f"{sample_id}.csv"


def process_sample(row: pd.Series, files: list[Path], data_root: Path) -> None:
    totals = compute_totals(row, files)
    if totals.contacts.sum() == 0:
        raise ValueError("no cis contacts found")
    cache = cache_path(data_root, row["sample_id"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "diagonal": np.arange(MAX_DIAGONAL + 1),
        "contacts": totals.contacts,
        "bin_pairs": totals.bin_pairs,
    }).to_csv(cache, index=False)


def summarise(table: pd.DataFrame, data_root: Path) -> None:
    """Fit every cached sample and write alpha_results.csv and ps_curves.csv."""
    results, curves = [], []
    for _, row in table.iterrows():
        cache = cache_path(data_root, row["sample_id"])
        if not cache.exists():
            continue
        counts = pd.read_csv(cache)
        curve = scaling_curve(counts["diagonal"].to_numpy(), counts["contacts"].to_numpy(),
                              counts["bin_pairs"].to_numpy())
        fit = fit_alpha(curve)
        results.append({
            "sample_id": row["sample_id"], "dataset": row["dataset"], "format": row["format"],
            "genome": row["genome"], "resolution_bp": RESOLUTION, **fit,
            "cis_contacts": int(counts["contacts"].sum()),
        })
        curves.append(curve.assign(sample_id=row["sample_id"]))

    if not results:
        print("\nNo samples have been computed yet, nothing to summarise.")
        return
    pd.DataFrame(results).to_csv(data_root / "alpha_results.csv", index=False)
    pd.concat(curves)[["sample_id", "distance_bp", "contact_probability"]].to_csv(
        data_root / "ps_curves.csv", index=False)
    print(f"\nSaved: {data_root / 'alpha_results.csv'} ({len(results)} samples)")
    print(f"Saved: {data_root / 'ps_curves.csv'}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--search", type=Path, action="append", required=True,
                        help="folder to look for input files in (repeat for several folders)")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"folder for the outputs (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--inputs", type=Path, default=INPUTS_PATH, help="input table")
    parser.add_argument("--only", nargs="+", metavar="SAMPLE_ID", help="process these samples only")
    parser.add_argument("--check", action="store_true",
                        help="only check that every input file can be found, then stop")
    parser.add_argument("--force", action="store_true", help="recompute samples that are cached")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    data_root = arguments.data_root.expanduser().resolve()
    all_samples = table = load_inputs(arguments.inputs)
    if arguments.only:
        unknown = sorted(set(arguments.only) - set(table["sample_id"]))
        if unknown:
            raise SystemExit(f"Not in {arguments.inputs.name}: {', '.join(unknown)}")
        table = table[table["sample_id"].isin(arguments.only)]

    print("Indexing search folders...")
    index = index_files(arguments.search)
    located, missing = {}, []
    for _, row in table.iterrows():
        try:
            located[row["sample_id"]] = locate_inputs(row, index)
        except FileNotFoundError as error:
            missing.append(f"{row['sample_id']}: {error}")
    print(f"{len(located)} of {len(table)} samples have their input files")
    for line in missing:
        print(f"  MISSING {line}")
    if arguments.check:
        return

    failed = []
    to_run = [r for _, r in table.iterrows() if r["sample_id"] in located]
    for number, row in enumerate(to_run, start=1):
        sample_id = row["sample_id"]
        label = f"[{number}/{len(to_run)}] {sample_id[:40]:40}"
        if cache_path(data_root, sample_id).exists() and not arguments.force:
            print(f"{label} cached")
            continue
        start = time.time()
        print(f"{label} ...", end=" ", flush=True)
        try:
            process_sample(row, located[sample_id], data_root)
            print(f"done in {time.time() - start:.0f} s")
        except Exception as error:  # one bad file must not stop an overnight run
            print(f"FAILED: {error}")
            failed.append(f"{sample_id}: {error}")

    summarise(all_samples, data_root)
    if missing or failed:
        print(f"\n{len(missing)} missing and {len(failed)} failed:")
        for line in missing + failed:
            print(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
