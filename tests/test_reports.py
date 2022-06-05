""" Tests for helper functions and profile classes in reports.py"""
import os
import math
import numbers
import tempfile
import filecmp
import numpy as np
import pandas as pd
import pytest
from src.pandahelper import reports

TEST_DATA_DIR = "tests/test_data"
TEST_DATA_FILE = "sample_collisions.csv"
TEST_DF = pd.read_csv(os.path.join(TEST_DATA_DIR, TEST_DATA_FILE))


def test_abbreviate_df_invalid_input():
    """Invalid data type inputs should raise Type Error"""
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
            reports.abbreviate_df(invalid)


def test_abbreviate_df_valid_output():
    """Valid input should lead to valid and consistent output"""
    output = reports.abbreviate_df(TEST_DF, 5, 5)
    assert isinstance(output, (pd.DataFrame, pd.Series))
    assert output.iloc[1]["LOCATION"] == "(40.7504, -73.985214)"
    assert output.iloc[8]["COLLISION_ID"] == 3676178


def test_abbreviate_df_most_plus_least_greater_sum():
    """Returns itself if first + last parameters exceede object length"""
    first = 300
    last = 300
    assert len(TEST_DF) < first + last
    assert TEST_DF.equals(reports.abbreviate_df(TEST_DF, first, last))


def test_abbreviate_df_negative_params():
    """Negative parameters raise Value Error"""
    invalid_params = [
        {"first": -5, "last": 5},
        {"first": 5, "last": -5},
        {"first": -5, "last": -5},
    ]
    for invalid in invalid_params:
        with pytest.raises(ValueError):
            reports.abbreviate_df(TEST_DF, **invalid)


def test_abbreviate_string_correct():
    """Correctly abbreviates string"""
    test_str = "test" * 10
    assert reports.abbreviate_string(test_str, limit=4) == "test"


def test_abbreviate_string_invalid():
    """Input of invalid data type raises Type Error"""
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
            reports.abbreviate_string(invalid)


def test_distribution_stats_valid():
    """Valid data returns dictionary containing all statistics with numeric values"""
    output = reports.distribution_stats(TEST_DF["NUMBER OF PERSONS INJURED"])
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


def test_distribution_stats_invalid():
    """Invalid data type raises Type error"""
    invalid_types = [
        TEST_DF,
        TEST_DF["BOROUGH"],  # pd.Series
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
            reports.distribution_stats(invalid)


def test_frequency_table_valid():
    """Frequency gives expected output"""
    d_index = ["0", "1", "2", "3", "8"]
    expected_data = {
        "Count": [160, 29, 7, 3, 1],
        "% of Total": ["80.00%", "14.50%", "3.50%", "1.50%", "0.50%"],
    }
    expected_df = pd.DataFrame(expected_data, index=d_index)
    table = reports.frequency_table(TEST_DF["NUMBER OF PERSONS INJURED"])
    assert isinstance(table, pd.DataFrame)
    assert expected_df.equals(table)


def test_frequency_table_invalid():
    """Invalid data type raises Type error"""
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
            reports.frequency_table(invalid)


def test_DataFrameReport():
    """Generated DataFrame report should match test report"""
    comparison_report = "test_df_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_report)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        reports.DataFrameProfile(TEST_DF, name="test_name").save_report(test_file)
        assert filecmp.cmp(compare_file, test_file, shallow=False)


def test_SeriesReport():
    """Generated Series report should match test report"""
    comparison_report = "test_series_injured_profile.txt"
    compare_file = os.path.join(TEST_DATA_DIR, comparison_report)
    with tempfile.TemporaryDirectory() as tmp:
        test_file = os.path.join(tmp, "temp.txt")
        reports.SeriesProfile(TEST_DF["NUMBER OF PERSONS INJURED"]).save_report(
            test_file
        )
        assert filecmp.cmp(compare_file, test_file, shallow=False)
