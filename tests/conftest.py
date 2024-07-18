"""Shared pytest fixtures for Panda-helper tests.

Note that fixtures with a package-scope are run once and then available as
cached value.
"""

import os
import numpy as np
import pandas as pd
import pytest
from .utils import make_category_data

TEST_DATA_DIR = "tests/test_data"
TEST_DATA_FILE = "sample_collisions.csv"
CAT_SERIES = "BOROUGH"
NUM_SERIES = "NUMBER OF PERSONS INJURED"


@pytest.fixture
def test_df(scope="package"):  # pylint: disable=W0613
    """Return test pd.DataFrame from sample of NYC collisions dataset."""
    return pd.read_csv(os.path.join(TEST_DATA_DIR, TEST_DATA_FILE))


@pytest.fixture
def cat_like_series(test_df, scope="package"):  # pylint: disable=W0613,W0621
    """Return categorical-like (object) series."""
    return test_df[CAT_SERIES]


@pytest.fixture
def num_series(test_df, scope="package"):  # pylint: disable=W0613,W0621
    """Return numerical series."""
    return test_df[NUM_SERIES]


@pytest.fixture
def non_series_invalid(scope="package"):  # pylint: disable=W0613
    """Provide list of non-pd.Series, invalid data types."""
    invalid_types = [
        None,
        False,
        True,
        0,
        -1,
        34,
        34.5,
        "data",
        {"data": "dictionary"},
        [1, 2, 3, 4],
        ["1", "2", "3", "4"],
        (1,),
        (("col_name", 3), ("col_name2", 4)),
        np.array([1, 2, 3]),
    ]
    return invalid_types


@pytest.fixture
def simple_df(scope="package"):  # pylint: disable=W0613
    """Return test pd.DataFrame with DatetimeIndex."""
    start = pd.Timestamp(year=1999, month=1, day=1)
    end = start + pd.Timedelta(hours=10)
    df = make_category_data("Springfield", start, end, freq="h")
    df = df.sample(frac=1, random_state=2)  # index is out of order
    return df


@pytest.fixture
def ts_timeindex(scope="package"):  # pylint: disable=W0613
    """Return pd.Series of type datetime64 with DatetimeIndex."""
    start = pd.Timestamp(year=1999, month=1, day=1)
    end = start + pd.Timedelta(hours=40)
    time_series = pd.Series(pd.date_range(start, end, freq="4h", inclusive="left"))
    index_end = start + pd.Timedelta(hours=10)
    time_series.index = pd.date_range(start, index_end, freq="h", inclusive="left")
    return time_series


@pytest.fixture
def cat_df(scope="package"):  # pylint: disable=W0613
    """Return pd.DataFrame with DatetimeIndex."""
    start = pd.Timestamp(year=1999, month=1, day=1)
    end = start + pd.Timedelta(days=365)
    delay = pd.Timedelta(days=180)
    c1 = make_category_data("Springfield", start, end, freq="h")
    c2 = make_category_data("Quahog", start + delay, end, freq="h")
    c3 = make_category_data("Park South", start, end, freq="2h")
    c4 = make_category_data("East Midtown", start, end, freq="4h")
    c5 = make_category_data("San Diego", start, end, freq="W")
    c6 = make_category_data("South Philadelphia", start, end, freq="MS")
    return pd.concat([c1, c2, c3, c4, c5, c6])
