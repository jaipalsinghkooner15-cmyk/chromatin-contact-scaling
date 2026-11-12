#!/usr/bin/env python3
"""
Download GEO files one at a time, compute their contact counts, then delete them.

Some inputs are too large to keep: the 29 samples marked download = geo in
data/input_files.csv come to about 182 GB. This script handles them like a
conveyor belt, so the disk only ever holds one of them:

    1. Download the file from GEO into the download folder.
    2. Check the download is complete (its size must match what GEO reports).
    3. Compute its contact counts with 02_compute_alpha.py, the same method as
       every other sample, and save them to scaling_cache.
    4. Delete the downloaded file.

Every step is written to download_log.csv in the download folder, so there is
a record of what came from where. Samples already in scaling_cache are skipped,
and an interrupted download resumes where it stopped, so an interrupted run can
simply be started again. At the end alpha_results.csv and ps_curves.csv are
rebuilt for every sample.

GEO keeps sample files under samples/GSMnnnnnnn and study files under
series/GSEnnnnnn; the address is built from the start of the file name.

Usage
    python preprocess/download_and_compute.py --download-dir "C:/.../Raw_data_HiC/geo_downloads"
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_PATH = REPO_ROOT / "data" / "input_files.csv"
DEFAULT_DATA_ROOT = Path.home() / "OneDrive" / "Documents" / "Dissertation"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
SPARE_SPACE = 5e9            # keep at least 5 GB free beyond the file being downloaded
ATTEMPTS = 5
CHUNK = 8 * 1024 * 1024
LOG_FIELDS = ["time", "sample_id", "file", "url", "size_gb", "status", "detail"]


def load_pipeline():
    path = REPO_ROOT / "preprocess" / "02_compute_alpha.py"
    spec = importlib.util.spec_from_file_location("compute_alpha", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geo_url(file_name: str) -> str:
    """GEO address of a supplementary file, from its GSM or GSE prefix."""
    accession = file_name.split("_")[0]
    if accession.startswith("GSM"):
        folder = "samples"
    elif accession.startswith("GSE"):
        folder = "series"
    else:
        raise ValueError(f"{file_name}: name does not start with a GSM or GSE number")
    return f"{GEO}/{folder}/{accession[:-3]}nnn/{accession}/suppl/{file_name}"


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(url: str, target: Path, expected: int) -> None:
    """Download to target, resuming a partial file, and check the final size."""
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        have = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:  # server ignored the resume request
                    have = 0
                with open(partial, "ab" if have else "wb") as handle:
                    start, done = time.time(), have
                    while block := response.read(CHUNK):
                        handle.write(block)
                        done += len(block)
                        rate = (done - have) / max(time.time() - start, 1e-6) / 1e6
                        print(f"\r    {done / expected:6.1%} of {expected / 1e9:.2f} GB "
                              f"at {rate:.1f} MB/s", end="", flush=True)
            print()
            if partial.stat().st_size == expected:
                partial.rename(target)
                return
            raise OSError(f"size {partial.stat().st_size} does not match GEO's {expected}")
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f"\n    attempt {attempt} of {ATTEMPTS} failed: {error}")
            if attempt == ATTEMPTS:
                raise
            time.sleep(30 * attempt)


def write_log(log_path: Path, **entry) -> None:
    new = not log_path.exists()
    with open(log_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"time": datetime.now().isoformat(timespec="seconds"), **entry})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--download-dir", type=Path, required=True,
                        help="folder for the one file being downloaded at a time")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"folder for scaling_cache and results (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--inputs", type=Path, default=INPUTS_PATH)
    arguments = parser.parse_args()

    pipeline = load_pipeline()
    data_root = arguments.data_root.expanduser().resolve()
    download_dir = arguments.download_dir.expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    log_path = download_dir / "download_log.csv"

    table = pipeline.load_inputs(arguments.inputs)
    if "download" not in table.columns:
        raise SystemExit(f"{arguments.inputs.name} has no download column")
    queue = [row for _, row in table[table["download"] == "geo"].iterrows()
             if not pipeline.cache_path(data_root, row["sample_id"]).exists()]
    print(f"{len(queue)} samples to download and compute")

    failed = []
    for number, row in enumerate(queue, start=1):
        sample_id, file_name = row["sample_id"], row["file"]
        url = geo_url(file_name)
        target = download_dir / file_name
        print(f"[{number}/{len(queue)}] {sample_id}: {file_name}")
        try:
            size = remote_size(url)
            free = shutil.disk_usage(download_dir).free
            if free < size + SPARE_SPACE:
                needed = (size + SPARE_SPACE) / 1e9
                raise OSError(f"only {free / 1e9:.0f} GB free, need {needed:.0f} GB")
            if not target.exists():
                download(url, target, size)
            write_log(log_path, sample_id=sample_id, file=file_name, url=url,
                      size_gb=round(size / 1e9, 3), status="downloaded", detail="")
            start = time.time()
            print("    computing contact counts ...", end=" ", flush=True)
            pipeline.process_sample(row, [target], data_root)
            print(f"done in {time.time() - start:.0f} s")
            target.unlink()
            write_log(log_path, sample_id=sample_id, file=file_name, url=url,
                      size_gb=round(size / 1e9, 3), status="computed and deleted", detail="")
        except Exception as error:  # one failure must not stop an overnight run
            print(f"    FAILED: {error}")
            failed.append(f"{sample_id}: {error}")
            write_log(log_path, sample_id=sample_id, file=file_name, url=url,
                      size_gb="", status="failed", detail=str(error))

    pipeline.summarise(table, data_root)
    leftovers = [p.name for p in download_dir.iterdir() if p.name != log_path.name]
    print(f"Download folder now holds: {', '.join(leftovers) if leftovers else 'only the log'}")
    if failed:
        print(f"\n{len(failed)} failed (run the same command again to retry):")
        for line in failed:
            print(f"  {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()
