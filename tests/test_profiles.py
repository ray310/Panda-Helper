""" Tests for profile classes in profiles.py."""

import filecmp
import os
import re
import tempfile
from datetime import datetime
import bs4
import numpy as np
import pandas as pd
import pytest
import pandahelper.profiles as php


TEST_DATA_DIR = "tests/test_data"
TEST_DATA_FILE = "sample_collisions.csv"
TEST_DF = pd.read_csv(os.path.join(TEST_DATA_DIR, TEST_DATA_FILE))
TEST_CAT_SERIES = TEST_DF["BOROUGH"]
TEST_NUM_SERIES = TEST_DF["NUMBER OF PERSONS INJURED"]


def test_dataframe_profile_valid():
    """Generated DataFrame profile should match test profile."""
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
            php.DataFrameProfile(TEST_DF, name=name).save_report(test_file)
            assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_dataframe_profile_invalid():
    """DataFrame profile should not accept invalid data types."""
    invalid_types = [
        TEST_CAT_SERIES,
        TEST_NUM_SERIES,
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
            php.DataFrameProfile(invalid)


def test_dataframe_profile_html():
    """Test html representation of DataFrameProfile."""
    profile = php.DataFrameProfile(TEST_DF)
    # fmt: off
    soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
    # fmt: on
    tables = soup.find_all("table")
    assert len(tables) == 3  # null_table
    assert len(tables[2].find_all("tr")) == 16  # 15 dist stats + head row
    first_td = tables[2].find("td")
    assert first_td["style"] == "font-family: monospace, monospace; text-align: left;"


def test_series_profile_text_valid_numerical_format():
    """Text version of SeriesProfile for numerical data matches test profile."""
    comparison_profile = "test_series_injured_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_profile)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.SeriesProfile(TEST_NUM_SERIES).save_report(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_series_profile_text_valid_object_format():
    """Text version of SeriesProfile for categorical data matches test profile."""
    comparison_profile = "test_series_borough_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_profile)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        php.SeriesProfile(TEST_CAT_SERIES).save_report(test_file)
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


def test_series_profile_invalid():
    """Series profile should not accept invalid data types."""
    invalid_types = [
        TEST_DF,
        "data",
        34,
        34.5,
        {"data": "dictionary"},
        [["col_name", 1], ["col_name2", 2]],
        (("col_name", 3), ("col_name2", 4)),
        np.array([1, 2, 3]),
        pd.arrays.IntervalArray([pd.Interval(0, 1), pd.Interval(1, 5)]),
        pd.Index(range(10)),
    ]
    for invalid in invalid_types:
        with pytest.raises(TypeError):
            php.SeriesProfile(invalid)


def test_series_profile_html():
    """Test html representation of SeriesProfile."""
    profile = php.SeriesProfile(TEST_NUM_SERIES)
    # fmt: off
    soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
    # fmt: on
    tables = soup.find_all("table")
    assert len(tables) == 3  # null_table
    assert len(tables[1].find_all("tr")) == 6  # freq table
    assert len(tables[2].find_all("tr")) == 16  # 15 dist stats + head row
    first_td = tables[2].find("td")
    assert first_td["style"] == "font-family: monospace, monospace; text-align: left;"


def test_series_profile_frequency_table():
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
        profile = php.SeriesProfile(TEST_DF["CRASH TIME"], freq_most_least=k)
        # fmt: off
        soup = bs4.BeautifulSoup(profile._repr_html_(), "html.parser")  # pylint: disable=W0212
        # fmt: on
        freq_table = soup.find_all("table")[1]
        assert len(freq_table.find_all("tr")) == v + 1  # +1 for header


def test_series_profile_frequency_table_invalid():
    """Invalid frequency table most_least tuples should raise ValueError."""
    invalid_tuples = [(0, -1), (-1, 0), (-1, -1)]
    with pytest.raises(ValueError):
        for invalid in invalid_tuples:
            php.SeriesProfile(TEST_DF["CRASH TIME"], freq_most_least=invalid)


def test_abbreviate_df_invalid_input():
    """Invalid data type inputs should raise Type Error."""
    invalid_types = [
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
            php._abbreviate_df(invalid)  # pylint: disable=W0212


def test_abbreviate_df_valid_output():
    """Valid input should lead to valid and consistent output."""
    output = php._abbreviate_df(TEST_DF, 5, 5)  # pylint: disable=W0212
    assert isinstance(output, (pd.DataFrame, pd.Series))
    assert output.iloc[1]["LOCATION"] == "(40.7504, -73.985214)"
    assert output.iloc[8]["COLLISION_ID"] == 3676178


def test_abbreviate_df_most_plus_least_greater_sum():
    """Returns itself if (first + last) parameters exceed object length."""
    first = 300
    last = 300
    assert len(TEST_DF) < first + last
    assert TEST_DF.equals(
        php._abbreviate_df(TEST_DF, first, last)  # pylint: disable=W0212
    )


def test_abbreviate_df_negative_params():
    """Negative parameters raise Value Error."""
    invalid_params = [
        {"first": -5, "last": 5},
        {"first": 5, "last": -5},
        {"first": -5, "last": -5},
    ]
    for invalid in invalid_params:
        with pytest.raises(ValueError):
            php._abbreviate_df(TEST_DF, **invalid)  # pylint: disable=W0212
