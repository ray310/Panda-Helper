"""Tests for profile classes in profiles.py."""

import filecmp
import os
import re
import sys
import tempfile
from datetime import datetime
import bs4
import numpy as np
import pandas as pd
import pandas.api.types as pat
import pytest
import pandahelper.profiles as php

TEST_DATA_DIR = "tests/test_data"  # needed
TEST_DATA_FILE = "sample_collisions.csv"


@pytest.mark.skipif(
    not ((3, 12) <= sys.version_info < (3, 13)), reason="Runs on Python 3.12"
)
def test_dataframe_profile_valid_312(test_df):
    """Generated DataFrame profile should match test profile (Python 3.12)."""
    compare_profile_name = "test_df_profile_name.txt"
    compare_profile_no_name = "test_df_profile_no_name.txt"
    compare_files = [
        os.path.join(TEST_DATA_DIR, compare_profile_name),
        os.path.join(TEST_DATA_DIR, compare_profile_no_name),
    ]
    names = ["test_name", ""]
    with tempfile.TemporaryDirectory() as tmp:
        for name, compare_file in zip(names, compare_files):
            test_file = os.path.join(tmp, "temp.txt")
            php.DataFrameProfile(test_df, name=name).save(test_file)
            assert filecmp.cmp(compare_file, test_file, shallow=False)


@pytest.mark.skipif(
    not ((3, 12) <= sys.version_info < (3, 13)), reason="Runs on Python 3.12"
)
def test_dataframe_time_profile_valid_312(cat_df):
    """Time-indexed DataFrame profile should match test profile (Python 3.12)."""
    compare_file = os.path.join(TEST_DATA_DIR, "test_df_time_profile.txt")
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.DataFrameProfile(cat_df).save(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_dataframe_profile_invalid(non_series_invalid, num_series, cat_like_series):
    """DataFrame profile should not accept invalid data types."""
    invalid_types = [*non_series_invalid, num_series, cat_like_series]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            php.DataFrameProfile(invalid)


def test_dataframe_profile_html(cat_df):
    """Test html representation of DataFrameProfile."""
    profile = php.DataFrameProfile(cat_df)
    # fmt: off
    soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
    # fmt: on
    tables = soup.find_all("table")
    assert len(tables) == 4
    assert len(tables[2].find_all("tr")) == 16  # 15 dist stats + head row
    first_td = tables[2].find("td")
    assert first_td["style"] == "font-family: monospace, monospace; text-align: left;"
    assert len(tables[3].find_all("tr")) == 3  # 2 deltas + head row


def test_series_profile_text_valid_numerical_format(num_series):
    """Text version of SeriesProfile for numerical data matches test profile."""
    comparison_profile = "test_series_injured_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_profile)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.SeriesProfile(num_series).save(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_series_profile_text_valid_object_format(cat_like_series):
    """Text version of SeriesProfile for categorical data matches test profile."""
    comparison_profile = "test_series_borough_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_profile)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.SeriesProfile(cat_like_series).save(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_series_profile_text_valid_time_format(cat_df):
    """Text version of SeriesProfile for time data matches test profile."""
    comparison_profile = "test_series_time_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_profile)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.SeriesProfile(cat_df["category"], time_index=True).save(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_series_profile_series_dtypes():
    """pd.Series should create SeriesProfile for allowed data types."""
    start = datetime(year=1999, month=1, day=1)
    dur = pd.Timedelta(days=10)
    end = start + dur
    series = [
        pd.Series(list(range(10))),  # int64
        pd.Series(list(np.arange(0.0, 10, 1))),  # float64
        pd.Series([bool(x % 2) for x in range(10)], name="bool"),  # bool
        pd.Series([complex(x, x) for x in range(10)]),  # complex128
        pd.Series(["Aa", "Bb", "Cc", "Dd", "Cc"]),  # object
        pd.Series(["Aa", "Bb", "Cc", "Dd", "Cc"], dtype="category"),  # category
        pd.Series(pd.date_range(start, end, freq="bh")),  # datetime64[ns]
        pd.Series([dur * x for x in range(10)]),  # timedelta64[ns]
        pd.Series([pd.Period(start + (dur * x), "M") for x in range(10)]),  # period[M]
        pd.Series(
            [pd.Interval(left=x, right=x + 2) for x in range(10)]
        ),  # interval[int64, right]
        pd.Series([None] * 10),  # object
        pd.Series([np.nan] * 10),  # float64
    ]
    for s in series:
        try:
            php.SeriesProfile(s)
        except Exception as ex:
            print(s)
            raise ex


def test_series_profile_singleton():
    """Profile should be generated for series with single item."""
    start = datetime(year=1999, month=1, day=1)
    dur = pd.Timedelta(days=0)
    end = start + dur
    series = [
        pd.Series(list(range(1))),  # int64
        pd.Series(list(np.arange(0.0, 1, 1))),  # float64
        pd.Series([bool(x % 2) for x in range(1)], name="bool"),  # bool
        pd.Series([complex(x, x) for x in range(1)]),  # complex128
        pd.Series(["Aa"]),  # object
        pd.Series(["Aa"], dtype="category"),  # category
        pd.Series(pd.date_range(start, end, freq="bh")),  # datetime64[ns]
        pd.Series([dur * x for x in range(1)]),  # timedelta64[ns]
        pd.Series([pd.Period(start + (dur * x), "M") for x in range(1)]),  # period[M]
        pd.Series(
            [pd.Interval(left=x, right=x + 2) for x in range(1)]
        ),  # interval[int64, right]
        pd.Series([None] * 1),  # object
        pd.Series([np.nan] * 1),  # float64
    ]
    for s in series:
        try:
            php.SeriesProfile(s)
        except Exception as ex:
            print(s)
            raise ex


def test_series_profile_complex():
    """Profile for complex series should contain and display imaginary component."""
    series = pd.Series([complex(x, x) for x in range(10)])
    profile = php.SeriesProfile(series)
    expected_stats = {
        "count": 10,
        "min": 0j,
        "max": (9 + 9j),
        "mean": (4.5 + 4.5j),
        "median": (4.5 + 4.5j),
    }
    assert profile.stats == expected_stats


def test_series_profile_complex_format():
    """Profile for complex series should contain and display imaginary component."""
    series = pd.Series([complex(x, x) for x in range(10)])
    profile = php.SeriesProfile(series)
    assert re.findall("mean\\s+[(]4.5[+]4.5j[)]", repr(profile))


def test_series_profile_invalid(non_series_invalid, test_df):
    """Series profile should not accept invalid data types."""
    invalid_series = [
        pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]),
        pd.Index(range(10)),
    ]
    invalid_types = [*non_series_invalid, test_df] + invalid_series
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            php.SeriesProfile(invalid)


def test_series_profile_html(cat_df):
    """Test html representation of SeriesProfile."""
    profile = php.SeriesProfile(cat_df["C"], time_index=True)
    # fmt: off
    soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
    # fmt: on
    tables = soup.find_all("table")
    assert len(tables) == 4
    assert len(tables[1].find_all("tr")) == 16  # freq table
    assert len(tables[2].find_all("tr")) == 16  # 15 dist stats + head row
    first_td = tables[2].find("td")
    assert first_td["style"] == "font-family: monospace, monospace; text-align: left;"
    assert len(tables[3].find_all("tr")) == 3  # 2 deltas + head row


def test_series_profile_frequency_table(test_df):
    """Valid values for frequency table should produce tables of desired length."""
    most_least_tuples = {
        (200, 200): 150,  # only 150 unique values
        (70, 70): 140,
        (1, 1): 2,
        (1, 0): 1,
        (0, 1): 1,
        (0, 0): 0,
    }
    for k, v in most_least_tuples.items():
        profile = php.SeriesProfile(test_df["CRASH TIME"], freq_most_least=k)
        # fmt: off
        soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
        # fmt: on
        freq_table = soup.find_all("table")[1]
        assert len(freq_table.find_all("tr")) == v + 1  # +1 for header


def test_series_profile_time_index_true(simple_df):
    """time_index=True calculates time diffs for Series with DateTimeIndex."""
    series = simple_df["category"]
    profile = php.SeriesProfile(series, time_index=True)
    assert pat.is_datetime64_any_dtype(series.index)
    assert profile.time_diffs.iloc[0] is pd.NaT
    assert all(profile.time_diffs[1:] == pd.Timedelta(hours=1))


def test_series_profile_time_index_false(simple_df):
    """time_index=False does not calculate time diffs for Series with DateTimeIndex."""
    series = simple_df["category"]
    profile = php.SeriesProfile(series, time_index=False)
    assert pat.is_datetime64_any_dtype(series.index)
    assert profile.time_diffs is None


def test_series_profile_ts_range_index_true(ts_timeindex):  # pylint: disable=W0621
    """time_index=True does not calculate time diffs for Series with RangeIndex."""
    series = ts_timeindex
    series.index = range(len(ts_timeindex))
    profile = php.SeriesProfile(series, time_index=True)
    assert not pat.is_datetime64_any_dtype(series.index)
    assert profile.time_diffs is None


def test_series_profile_both_time_index_false(ts_timeindex):  # pylint: disable=W0621
    """SeriesProfile should have time diffs from series, (not index).

    Given for Series(datetime64) with TimeIndex and time_index=False.
    """
    profile = php.SeriesProfile(ts_timeindex, time_index=False)
    assert pat.is_datetime64_any_dtype(ts_timeindex.index)
    assert profile.time_diffs.iloc[0] is pd.NaT
    assert all(profile.time_diffs[1:] == pd.Timedelta(hours=4))


def test_series_profile_both_time_index_true(ts_timeindex):  # pylint: disable=W0621
    """SeriesProfile should have time diffs from index, (not series).

    Given for Series(datetime64) with TimeIndex and time_index=True.
    """
    profile = php.SeriesProfile(ts_timeindex, time_index=True)
    assert pat.is_datetime64_any_dtype(ts_timeindex.index)
    assert profile.time_diffs.iloc[0] is pd.NaT
    assert all(profile.time_diffs[1:] == pd.Timedelta(hours=1))


def test_series_profile_frequency_table_invalid(test_df):
    """Invalid frequency table most_least tuples should raise ValueError."""
    invalid_tuples = [(0, -1), (-1, 0), (-1, -1)]
    with pytest.raises(ValueError):
        for invalid in invalid_tuples:
            php.SeriesProfile(test_df["CRASH TIME"], freq_most_least=invalid)


def test_abbreviate_df_invalid_input(non_series_invalid):
    """Invalid data type inputs should raise Type Error."""
    for invalid in non_series_invalid:
        with pytest.raises(TypeError):
            php._abbreviate_df(invalid)  # pylint: disable=W0212


def test_abbreviate_df_valid_output(test_df):
    """Valid input should lead to valid and consistent output."""
    output = php._abbreviate_df(test_df, 5, 5)  # pylint: disable=W0212
    assert isinstance(output, (pd.DataFrame, pd.Series))
    assert output.iloc[1]["LOCATION"] == "(40.7504, -73.985214)"
    assert output.iloc[8]["COLLISION_ID"] == 3676178


def test_abbreviate_df_most_plus_least_greater_sum(test_df):
    """Returns itself if (first + last) parameters exceed object length."""
    first = 300
    last = 300
    assert len(test_df) < first + last
    assert test_df.equals(
        php._abbreviate_df(test_df, first, last)  # pylint: disable=W0212
    )


def test_abbreviate_df_negative_params(test_df):
    """Negative parameters raise Value Error."""
    invalid_params = [
        {"first": -5, "last": 5},
        {"first": 5, "last": -5},
        {"first": -5, "last": -5},
    ]
    for invalid in invalid_params:
        with pytest.raises(ValueError):
            php._abbreviate_df(test_df, **invalid)  # pylint: disable=W0212
