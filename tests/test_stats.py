"""Tests for stats functions in stats.py."""

import math
import numbers
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import pandahelper.stats as phs


@pytest.fixture
def full_stats(scope="module"):  # pylint: disable=W0613
    """Return full list of distribution statistics."""
    stats = [
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
    return stats


def test_frequency_table_valid(num_series):
    """Frequency gives expected output."""
    d_index = ["0", "1", "2", "3", "8"]
    expected_data = {
        "Count": [160, 29, 7, 3, 1],
        "% of Total": ["80.00%", "14.50%", "3.50%", "1.50%", "0.50%"],
    }
    expected_df = pd.DataFrame(expected_data, index=d_index)
    table = phs.frequency_table(num_series)
    assert isinstance(table, pd.DataFrame)
    assert expected_df.equals(table)


def test_frequency_table_invalid(non_series_invalid, test_df):
    """Non-pd.Series data types raises Type error."""
    invalid_types = [*non_series_invalid, test_df]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs.frequency_table(invalid)


def test_abbreviate_string_correct():
    """Correctly abbreviates string."""
    test_str = "test" * 10
    # fmt: off
    assert phs._abbreviate_string(test_str, limit=4) == "test"  # pylint: disable=W0212
    # fmt: on


def test_abbreviate_string_invalid(non_series_invalid, test_df, cat_like_series):
    """Input of invalid data type raises Type Error."""
    non_series_invalid.remove("data")
    invalid_types = [*non_series_invalid, test_df, cat_like_series]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs._abbreviate_string(invalid)  # pylint: disable=W0212


def test_distribution_stats(num_series, full_stats):  # pylint: disable=W0621
    """distribution_stats returns 1-column pd.DataFrame with correctly ordered stats."""
    output = phs.distribution_stats(num_series)
    assert isinstance(output, pd.DataFrame)
    assert len(output.keys()) == 1
    assert output.index.to_list() == full_stats


def test_dist_stats_dict(num_series):
    """Return dictionary containing all statistics with numeric values."""
    output = phs.dist_stats_dict(num_series)
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


def test_dist_stats_dict_int_float(full_stats):  # pylint: disable=W0621
    """Int64 and float64 series should return all summary stats."""
    series = [
        pd.Series(list(range(10))),  # int64
        pd.Series(list(np.arange(0.0, 10, 1))),  # float64
        pd.Series([np.nan] * 10),  # float64
    ]
    for s in series:
        ds = phs.dist_stats_dict(s)
        assert list(ds.keys()) == full_stats  # also checks order


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_dist_stats_dict_infinity(full_stats):  # pylint: disable=W0621
    """Series should return all summary stats though some are nan."""
    series = pd.Series([float("inf"), 1, 2, 3])
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == full_stats  # also checks order


def test_dist_stats_dict_bool():
    """Boolean series should return below summary stats."""
    series = pd.Series([bool(x % 2) for x in range(10)], name="bool")  # bool
    expected_stats = [
        "count",
        "min",
        "max",
        "mean",
    ]
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_complex():
    """Complex series should return below summary stats."""
    series = pd.Series([complex(x, x) for x in range(10)])  # complex128
    expected_stats = [
        "count",
        "min",
        "max",
        "mean",
        "median",
    ]
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_timestamp():
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
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_timedelta():
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
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_period():
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
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_interval():
    """Pd.Interval series should return below summary stats."""
    series = pd.Series(
        [pd.Interval(left=x, right=x + 2) for x in range(10)]
    )  # interval[int64, right]
    expected_stats = [
        "count",
        "min",
        "max",
    ]
    ds = phs.dist_stats_dict(series)
    assert list(ds.keys()) == expected_stats  # also checks order


def test_dist_stats_dict_invalid(non_series_invalid, test_df, cat_like_series):
    """Invalid data type raises Type error."""
    start = datetime(year=1999, month=1, day=1)
    end = datetime(year=1999, month=1, day=11)
    invalid_types = [*non_series_invalid, test_df, cat_like_series]
    invalid_types += [
        pd.Series(["Aa", "Bb", "Cc", "Dd", "Cc"], dtype="category"),
        pd.DatetimeIndex(pd.date_range(start, end, freq="bh")),
    ]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            phs.dist_stats_dict(invalid)
