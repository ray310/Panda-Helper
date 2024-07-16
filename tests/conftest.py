"""Shared pytest fixtures for Panda-helper tests.

Note that fixtures with a package-scope are run once and then available as
cached value.
"""

from datetime import datetime
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
def cat_df(scope="package"):  # pylint: disable=W0613
    """Return test pd.DataFrame."""
    start = datetime(year=1999, month=1, day=1, hour=0, minute=0)
    end = start + pd.Timedelta(hours=10)
    df = make_category_data("Springfield", start, end, freq="h")
    df = df.sample(frac=1, random_state=2)  # index is out of order
    return df


@pytest.fixture
def test_df(scope="package"):  # pylint: disable=W0613
    """Return test pd.DataFrame."""
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
