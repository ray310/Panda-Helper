"""Tests for stats functions in stats.py."""

import math
import numbers
import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import pandahelper.stats as phs

TEST_DATA_DIR = "tests/test_data"
TEST_DATA_FILE = "sample_collisions.csv"
TEST_DF = pd.read_csv(os.path.join(TEST_DATA_DIR, TEST_DATA_FILE))
TEST_CAT_SERIES = TEST_DF["BOROUGH"]
TEST_NUM_SERIES = TEST_DF["NUMBER OF PERSONS INJURED"]


def test_frequency_table_valid():
    """Frequency gives expected output."""
    d_index = ["0", "1", "2", "3", "8"]
    expected_data = {
        "Count": [160, 29, 7, 3, 1],
        "% of Total": ["80.00%", "14.50%", "3.50%", "1.50%", "0.50%"],
    }
    expected_df = pd.DataFrame(expected_data, index=d_index)
    table = phs.frequency_table(TEST_DF["NUMBER OF PERSONS INJURED"])
    assert isinstance(table, pd.DataFrame)
    assert expected_df.equals(table)


def test_frequency_table_invalid():
    """Non-pd.Series data types raises Type error."""
    invalid_types = [
        TEST_DF,
        "data",
        34,
        34.5,
        {"data": "dictionary"},
        [["col_name", 1], ["col_name2", 2]],
        (("col_name", 3), ("col_name2", 4)),
        np.array([1, 2, 3]),
    ]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs.frequency_table(invalid)


def test_abbreviate_string_correct():
    """Correctly abbreviates string."""
    test_str = "test" * 10
    # fmt: off
    assert phs._abbreviate_string(test_str, limit=4) == "test"  # pylint: disable=W0212
    # fmt: on


def test_abbreviate_string_invalid():
    """Input of invalid data type raises Type Error."""
    invalid_types = [
        34,
        34.5,
        {"data": "dictionary"},
        [["col_name", 1], ["col_name2", 2]],
        (("col_name", 3), ("col_name2", 4)),
        np.array([1, 2, 3]),
        TEST_DF,
    ]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs._abbreviate_string(invalid)  # pylint: disable=W0212


def test_distribution_stats():
    """Return dictionary containing all statistics with numeric values."""
    output = phs.distribution_stats(TEST_DF["NUMBER OF PERSONS INJURED"])
    assert isinstance(output, dict)
    stats = {
        "count": 200,
        "1%": 0,
        "5%": 0.0,
        "25%": 0.0,
        "50%": 0.0,
        "75%": 0.0,
        "95%": 2.0,
        "99%": 3.0,
        "max": 8,
        "median": 0.0,
        "mean": 0.3,
        "median absolute deviation": 0.0,
        "standard deviation": 0.8082489292651694,
        "skew": 5.341196456568395,
    }
    for key, val in stats.items():
        assert key in output
        assert isinstance(output[key], numbers.Number)
        assert math.isclose(output[key], val)


def test_distribution_stats_int_float():
    """Int64 and float64 series should return all summary stats."""
    series = [
        pd.Series(list(range(10))),  # int64
        pd.Series(list(np.arange(0.0, 10, 1))),  # float64
        pd.Series([np.nan] * 10),  # float64
    ]
    expected_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "mean",
        "standard deviation",
        "median",
        "median absolute deviation",
        "skew",
    ]
    for s in series:
        ds = phs.distribution_stats(s)
        assert list(ds.keys()) == expected_stats  # also checks order


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_distribution_stats_infinity():
    """Series should return all summary stats though some are nan."""
    series = pd.Series([float("inf"), 1, 2, 3])
    expected_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "mean",
        "standard deviation",
        "median",
        "median absolute deviation",
        "skew",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_bool():
    """Boolean series should return below summary stats."""
    series = pd.Series([bool(x % 2) for x in range(10)], name="bool")  # bool
    expected_stats = [
        "count",
        "min",
        "max",
        "mean",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_complex():
    """Complex series should return below summary stats."""
    series = pd.Series([complex(x, x) for x in range(10)])  # complex128
    expected_stats = [
        "count",
        "min",
        "max",
        "mean",
        "median",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_timestamp():
    """Pd.Timestamp series should return below summary stats."""
    start = datetime(year=1999, month=1, day=1)
    end = datetime(year=1999, month=1, day=11)
    series = pd.Series(pd.date_range(start, end, freq="bh"))  # datetime64[ns]
    expected_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "mean",
        "standard deviation",
        "median",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_timedelta():
    """Pd.Timedelta series should return below summary stats."""
    dur = pd.Timedelta(days=10)
    series = pd.Series([dur * x for x in range(10)])  # timedelta64[ns]
    expected_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "mean",
        "standard deviation",
        "median",
        "median absolute deviation",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_period():
    """Pd.Period series should return below summary stats."""
    start = datetime(year=1999, month=1, day=1)
    dur = pd.Timedelta(days=10)
    series = pd.Series(
        [pd.Period(start + (dur * x), "M") for x in range(10)]
    )  # period[M]
    expected_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "median",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_interval():
    """Pd.Interval series should return below summary stats."""
    series = pd.Series(
        [pd.Interval(left=x, right=x + 2) for x in range(10)]
    )  # interval[int64, right]
    expected_stats = [
        "count",
        "min",
        "max",
    ]
    ds = phs.distribution_stats(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_distribution_stats_invalid():
    """Invalid data type raises Type error."""
    start = datetime(year=1999, month=1, day=1)
    end = datetime(year=1999, month=1, day=11)
    invalid_types = [
        TEST_DF,
        TEST_CAT_SERIES,
        pd.Series(["Aa", "Bb", "Cc", "Dd", "Cc"], dtype="category"),
        pd.DatetimeIndex(pd.date_range(start, end, freq="bh")),
        "data",
        34,
        34.5,
        {"data": "dictionary"},
        [["col_name", 1], ["col_name2", 2]],
        (("col_name", 3), ("col_name2", 4)),
        np.array([1, 2, 3]),
    ]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs.distribution_stats(invalid)
