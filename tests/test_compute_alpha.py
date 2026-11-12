"""
Tests for preprocess/02_compute_alpha.py.

Synthetic contact maps are built with a known scaling exponent, written in every
format the pipeline reads, and the pipeline must recover the exponent and give
identical counts whatever the format.

Run with:  python -m pytest tests
Needs:     pytest, hictkpy, pyarrow and cooler (pyarrow and cooler only to write
           the test files; the pipeline itself does not use them)
"""

import gzip
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "preprocess" / "02_compute_alpha.py"

spec = importlib.util.spec_from_file_location("compute_alpha", SCRIPT)
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

RES = ca.RESOLUTION
TRUE_ALPHA = 1.2
# 22 autosomes plus chrX: two large chromosomes that reach past the 10 Mb fit
# window, 20 small ones, and one sex chromosome that must be ignored.
SIZES = {"chr1": 12_000_000, "chr2": 10_000_000}
SIZES.update({f"chr{n}": 2_000_000 for n in range(3, 23)})
SIZES["chrX"] = 3_000_000


def bin_offsets(sizes):
    starts, total = {}, 0
    for chrom, size in sizes.items():
        starts[chrom] = total
        total += -(-size // RES)
    return starts


def pixels(expected_count, sizes=SIZES):
    """Upper-triangle cis pixels with count = expected_count(d) for every chromosome."""
    starts = bin_offsets(sizes)
    frames = []
    for chrom, size in sizes.items():
        n = -(-size // RES)
        i, j = np.triu_indices(n)
        keep = (j - i) <= ca.MAX_DIAGONAL
        i, j = i[keep], j[keep]
        counts = expected_count(j - i)
        nonzero = counts > 0
        frames.append(pd.DataFrame({
            "bin1_id": i[nonzero] + starts[chrom],
            "bin2_id": j[nonzero] + starts[chrom],
            "count": counts[nonzero].astype(np.int32),
        }))
    return pd.concat(frames, ignore_index=True)


def dense_power_law(d):
    """Deterministic counts that follow d^-alpha exactly enough to recover alpha."""
    return np.round(1e6 * np.maximum(d, 1) ** -TRUE_ALPHA)


def sparse_power_law(seed=0, scale=3.0):
    """Poisson counts with mean scale * d^-alpha: mostly zeros at long range."""
    rng = np.random.default_rng(seed)
    return lambda d: rng.poisson(scale * np.maximum(d, 1) ** -TRUE_ALPHA)


def write_cool(path, table):
    import hictkpy
    writer = hictkpy.cooler.FileWriter(str(path), SIZES, RES, tmpdir=str(path.parent))
    writer.add_pixels(table)
    writer.finalize()


def write_hic(path, table):
    import hictkpy
    writer = hictkpy.hic.FileWriter(str(path), SIZES, RES, tmpdir=str(path.parent))
    writer.add_pixels(table)
    writer.finalize()


def totals_for(file_format, files, genome=""):
    row = pd.Series({"format": file_format, "genome": genome})
    return ca.compute_totals(row, files)


def alpha_from(totals):
    curve = ca.scaling_curve(np.arange(ca.MAX_DIAGONAL + 1), totals.contacts, totals.bin_pairs)
    return ca.fit_alpha(curve)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dense")
    table = pixels(dense_power_law)
    write_cool(folder / "dense.cool", table)
    write_hic(folder / "dense.hic", table)
    return folder, table


def test_recovers_known_alpha_from_cool_and_hic(dense):
    folder, _ = dense
    for name, file_format in [("dense.cool", "cool"), ("dense.hic", "hic")]:
        fit = alpha_from(totals_for(file_format, [folder / name]))
        assert fit["alpha"] == pytest.approx(TRUE_ALPHA, abs=0.005), name
        assert fit["r_squared"] > 0.999
        assert fit["empty_fit_bins"] == 0


def test_mcool_matches_cool(dense):
    import cooler
    folder, _ = dense
    cooler.zoomify_cooler(str(folder / "dense.cool"), str(folder / "dense.mcool"),
                          resolutions=[RES, 2 * RES], chunksize=1_000_000)
    from_cool = totals_for("cool", [folder / "dense.cool"])
    from_mcool = totals_for("mcool", [folder / "dense.mcool"])
    np.testing.assert_array_equal(from_cool.contacts, from_mcool.contacts)
    np.testing.assert_array_equal(from_cool.bin_pairs, from_mcool.bin_pairs)


def test_sex_chromosomes_are_ignored(dense):
    folder, _ = dense
    totals = totals_for("cool", [folder / "dense.cool"])
    autosome_bins = sum(-(-s // RES) for c, s in SIZES.items() if c != "chrX")
    assert totals.bin_pairs[0] == autosome_bins


def test_hic_per_chromosome_matches_single_file(dense, tmp_path):
    folder, table = dense
    starts = bin_offsets(SIZES)
    files = []
    for n in range(1, 23):
        chrom = f"chr{n}"
        first, last = starts[chrom], starts[chrom] + -(-SIZES[chrom] // RES)
        part = table[(table["bin1_id"] >= first) & (table["bin1_id"] < last)]
        path = tmp_path / f"sample.{chrom}.hic"
        write_hic(path, part)
        files.append(path)
    split = totals_for("hic_per_chromosome", files)
    whole = totals_for("hic", [folder / "dense.hic"])
    np.testing.assert_array_equal(split.contacts, whole.contacts)
    np.testing.assert_array_equal(split.bin_pairs, whole.bin_pairs)


def test_zeros_are_counted_so_sparse_data_are_not_biased():
    """The fixed bug: averaging only non-zero pixels flattens sparse curves."""
    table = pixels(sparse_power_law(seed=1, scale=3.0))
    totals = ca.DistanceTotals()
    for size in (s for c, s in SIZES.items() if c != "chrX"):
        totals.add_chromosome(-(-size // RES))
    starts = bin_offsets(SIZES)
    autosomes = table["bin1_id"] < starts["chrX"]
    diagonals = (table["bin2_id"] - table["bin1_id"])[autosomes].to_numpy()
    counts = table["count"][autosomes].to_numpy().astype(float)
    totals.add_contacts(diagonals, counts)

    alpha_with_zeros = alpha_from(totals)["alpha"]
    assert alpha_with_zeros == pytest.approx(TRUE_ALPHA, abs=0.05)

    nonzero_pixels = np.bincount(diagonals, minlength=ca.MAX_DIAGONAL + 1).astype(float)
    alpha_without_zeros = alpha_from(ca.DistanceTotals(totals.contacts, nonzero_pixels))["alpha"]
    assert alpha_without_zeros < TRUE_ALPHA - 0.3


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sparse")
    table = pixels(sparse_power_law(seed=2, scale=50.0))
    write_cool(folder / "sparse.cool", table)
    return folder, table


def pairs_from(table, seed=3):
    """One contact pair per count, at random positions inside each bin."""
    names = list(SIZES)
    starts = bin_offsets(SIZES)
    first_bins = np.array([starts[c] for c in names])
    rng = np.random.default_rng(seed)
    i = np.repeat(table["bin1_id"].to_numpy(), table["count"].to_numpy())
    j = np.repeat(table["bin2_id"].to_numpy(), table["count"].to_numpy())
    chrom_index = np.searchsorted(first_bins, i, side="right") - 1
    chrom = np.array(names)[chrom_index]
    pos1 = (i - first_bins[chrom_index]) * RES + rng.integers(1, RES, len(i))
    pos2 = (j - first_bins[chrom_index]) * RES + rng.integers(1, RES, len(j))
    limits = np.array([SIZES[c] for c in names])[chrom_index]
    return chrom, np.minimum(pos1, limits), np.minimum(pos2, limits)


def test_4dn_pairs_match_cool(sparse, tmp_path):
    folder, table = sparse
    chrom, pos1, pos2 = pairs_from(table)
    path = tmp_path / "sample.pairs.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("## pairs format v1.0\n")
        for name, size in SIZES.items():
            handle.write(f"#chromsize: {name} {size}\n")
        handle.write("#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 pair_type\n")
        for k in range(len(chrom)):
            handle.write(f"r{k}\t{chrom[k]}\t{pos1[k]}\t{chrom[k]}\t{pos2[k]}\t+\t-\tUU\n")
    from_pairs = totals_for("pairs_4dn", [path])
    from_cool = totals_for("cool", [folder / "sparse.cool"])
    np.testing.assert_array_equal(from_pairs.contacts, from_cool.contacts)
    np.testing.assert_array_equal(from_pairs.bin_pairs, from_cool.bin_pairs)


def test_hicpro_validpairs_match_cool(sparse, tmp_path, monkeypatch):
    folder, table = sparse
    monkeypatch.setitem(ca.GENOME_SIZES, "test", SIZES)
    chrom, pos1, pos2 = pairs_from(table, seed=4)
    path = tmp_path / "sample_allValidPairs.txt.gz"
    with gzip.open(path, "wt") as handle:
        for k in range(len(chrom)):
            handle.write(f"r{k}\t{chrom[k]}\t{pos1[k]}\t+\t{chrom[k]}\t{pos2[k]}\t-\t300\n")
        handle.write("rx\tchr1\t5000\t+\tchr2\t5000\t-\t300\n")  # trans pair, ignored
    from_pairs = totals_for("validpairs_hicpro", [path], genome="test")
    from_cool = totals_for("cool", [folder / "sparse.cool"])
    np.testing.assert_array_equal(from_pairs.contacts, from_cool.contacts)


def test_pairs_beyond_chromosome_end_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setitem(ca.GENOME_SIZES, "test", SIZES)
    path = tmp_path / "wrong_genome.txt"
    path.write_text("r1\tchr3\t100\t+\tchr3\t2500000\t-\t300\n")
    with pytest.raises(ValueError, match="beyond chromosome ends"):
        totals_for("validpairs_hicpro", [path], genome="test")


def write_hicpro_matrix(path, shift=0, chrm_bins=0):
    """Dense near-diagonal band on hg19 chr21 and chr22, genome-wide 1-based bins.

    chrm_bins adds a chrM-like chromosome after chrY, as HiC-Pro often does.
    """
    sizes = ca.GENOME_SIZES["hg19"]
    starts = bin_offsets(sizes)
    rows = []
    for chrom in ("chr21", "chr22"):
        n = -(-sizes[chrom] // RES)
        for d in range(0, 30):
            i = np.arange(n - d) + starts[chrom] + 1 + shift
            rows.append(pd.DataFrame({"i": i, "j": i + d, "count": 30 - d}))
    if chrm_bins:
        first = sum(-(-s // RES) for s in sizes.values()) + 1
        i = np.arange(first, first + chrm_bins)
        rows.append(pd.DataFrame({"i": i, "j": i, "count": 500}))
    pd.concat(rows).to_csv(path, sep="\t", header=False, index=False)
    lengths = [-(-sizes[c] // RES) for c in ("chr21", "chr22")]
    return np.array([sum(n - d for n in lengths) * (30 - d) for d in range(30)])


@pytest.mark.parametrize("chrm_bins", [0, 2])
def test_hicpro_matrix_uses_the_right_bin_layout(tmp_path, chrm_bins):
    path = tmp_path / "sample_10k.raw.matrix.txt"
    expected = write_hicpro_matrix(path, chrm_bins=chrm_bins)
    totals = totals_for("matrix_hicpro", [path], genome="hg19")
    np.testing.assert_array_equal(totals.contacts[:30], expected)
    assert totals.contacts[30:].sum() == 0


@pytest.mark.parametrize("shift", [1, 3])
def test_hicpro_matrix_with_wrong_layout_is_rejected(tmp_path, shift):
    path = tmp_path / "shifted.matrix.txt"
    write_hicpro_matrix(path, shift=shift)
    with pytest.raises(ValueError, match="bin layout does not match"):
        totals_for("matrix_hicpro", [path], genome="hg19")


def test_fit_needs_enough_points():
    curve = pd.DataFrame({"distance_bp": [2e5, 3e5], "contacts": [5, 4],
                          "bin_pairs": [10, 10], "contact_probability": [0.5, 0.4]})
    with pytest.raises(ValueError, match="usable points"):
        ca.fit_alpha(curve)


def test_input_table_problems_are_reported(tmp_path):
    path = tmp_path / "inputs.csv"
    pd.DataFrame({
        "sample_id": ["a", "a", "b"], "dataset": ["x"] * 3, "file": ["f"] * 3,
        "format": ["hic", "hic", "validpairs_hicpro"], "genome": ["", "", "mm10"],
    }).to_csv(path, index=False)
    with pytest.raises(SystemExit) as error:
        ca.load_inputs(path)
    assert "duplicate sample_id: a" in str(error.value)
    assert "genome must be hg19 or hg38 for: b" in str(error.value)


def test_locate_accepts_identical_copies_and_rejects_different_versions(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "same.hic").write_text("abc")
    (tmp_path / "two" / "same.hic").write_text("abc")
    (tmp_path / "one" / "clash.hic").write_text("abc")
    (tmp_path / "two" / "clash.hic").write_text("abcdef")
    index = ca.index_files([tmp_path])
    assert ca.locate("same.hic", index).name == "same.hic"
    with pytest.raises(FileNotFoundError, match="different versions"):
        ca.locate("clash.hic", index)
    with pytest.raises(FileNotFoundError, match="not found"):
        ca.locate("absent.hic", index)


def test_full_run_writes_results_and_reuses_the_cache(dense, tmp_path):
    folder, _ = dense
    inputs = tmp_path / "inputs.csv"
    pd.DataFrame({
        "sample_id": ["S1", "S2"], "dataset": ["synthetic"] * 2,
        "file": ["dense.cool", "dense.hic"], "format": ["cool", "hic"], "genome": ["", ""],
    }).to_csv(inputs, index=False)
    command = [sys.executable, str(SCRIPT), "--inputs", str(inputs),
               "--search", str(folder), "--data-root", str(tmp_path)]

    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    results = pd.read_csv(tmp_path / "alpha_results.csv")
    assert list(results["sample_id"]) == ["S1", "S2"]
    assert np.allclose(results["alpha"], TRUE_ALPHA, atol=0.005)
    assert (tmp_path / "ps_curves.csv").exists()

    second = subprocess.run(command, capture_output=True, text=True)
    assert second.stdout.count("cached") == 2

    only_one = subprocess.run(command + ["--only", "S1", "--force"], capture_output=True, text=True)
    assert only_one.returncode == 0
    assert list(pd.read_csv(tmp_path / "alpha_results.csv")["sample_id"]) == ["S1", "S2"]
