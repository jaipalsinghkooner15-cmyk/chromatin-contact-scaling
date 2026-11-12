"""Tests for analysis/make_figures.py: the figures must refuse numbers that disagree with the stats."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "analysis" / "make_figures.py"
spec = importlib.util.spec_from_file_location("make_figures", PATH)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def tiny_tables(median_delta: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = pd.DataFrame({
        "unit": ["t1", "t2", "t3", "c1", "c2", "c3"],
        "group": ["tumour"] * 3 + ["cancer cell line"] * 3,
        "cancer_type": ["COAD"] * 6,
        "alpha": [-1.10, -1.12, -1.14, -1.30, -1.28, -1.26],
    })
    rows = [{"analysis": figures.MAIN, "comparison": c, "n_tumour": 3, "n_cell_line": 3,
             "median_delta": median_delta} for c in ("overall", "COAD")]
    return units, pd.DataFrame(rows)


def test_matching_tables_pass():
    units, stats_table = tiny_tables(median_delta=-1.28 - -1.12)
    check = figures.check_against_stats(units, stats_table)
    assert check["agree"].all()
    assert len(check) == 6  # n tumour, n cell line, median delta, for overall and COAD


def test_disagreeing_tables_stop_the_figures():
    units, stats_table = tiny_tables(median_delta=-0.5)
    with pytest.raises(SystemExit):
        figures.check_against_stats(units, stats_table)
