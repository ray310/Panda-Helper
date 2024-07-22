"""Tests for functions in times.py."""

import numpy as np
import pandas as pd
import pytest
import pandahelper.times as pht
from .utils import make_category_data


def test_time_diffs(simple_df):
    """time_diffs should work on shuffled pd.Series or Index of timestamps."""
    valid = [simple_df.index, pd.Series(simple_df.index)]
    for v in valid:
        result = pht.time_diffs(v)
        assert result.iloc[0] is pd.NaT
        assert all(result[1:] == pd.Timedelta(hours=1))


def test_time_diffs_exception():
    """Non-datetime64 pd.Series raises exception."""
    invalid = [pd.Series(list(range(5))), pd.Series([pd.Timedelta(hours=1)] * 2)]
    for tipo in invalid:
        with pytest.raises(TypeError):
            pht.time_diffs(tipo)


def test_time_diffs_index(simple_df):
    """time_diffs_index should work on shuffled pd.Series or pd.DataFrame."""
    # test DF
    df_result = pht.time_diffs_index(simple_df)
    assert df_result.iloc[0] is pd.NaT
    assert all(df_result[1:] == pd.Timedelta(hours=1))
    # test Series
    series_result = pht.time_diffs_index(simple_df["B"])
    assert series_result.iloc[0] is pd.NaT
    assert all(series_result[1:] == pd.Timedelta(hours=1))


def test_time_diffs_index_exception():
    """pd.DataFrame and pd.Series without time index raise exception."""
    data = {"A": list(range(5))}
    dtypes = [pd.DataFrame(data), pd.Series(data)]
    for tipo in dtypes:
        with pytest.raises(TypeError) as exc:
            pht.time_diffs_index(tipo)
        assert str(pd.DatetimeIndex) in str(exc)


def test_id_gaps_index(ts_timeindex):
    """id_gap_index returns expected gap from time-Series with DatetimeIndex."""
    result = pht.id_gaps_index(
        ts_timeindex, pd.Timedelta(minutes=59, microseconds=999999)
    )
    expected = pd.DataFrame(
        [pd.Timedelta(hours=1)] * 9,
        index=pd.date_range(pd.Timestamp(1999, 1, 1, 1), periods=9, freq="h"),
        columns=["diffs"],
    )
    pd.testing.assert_frame_equal(expected, result, check_index_type=True)


def test_id_gaps_index_no_gaps(ts_timeindex):
    """id_gap_index returns empty Dataframe when threshold exceeds diffs."""
    result = pht.id_gaps_index(ts_timeindex, pd.Timedelta(minutes=60, microseconds=1))
    assert len(result) == 0


def test_id_gaps_(ts_timeindex):
    """id_gap returns expected gap from time-Series with DatetimeIndex."""
    result = pht.id_gaps(
        ts_timeindex, pd.Timedelta(hours=3, minutes=59, microseconds=999999)
    )
    expected = pd.DataFrame(
        [pd.Timedelta(hours=4)] * 9,
        index=pd.date_range(pd.Timestamp(1999, 1, 1, 4), periods=9, freq="4h"),
        columns=["diffs"],
    )
    expected.index.freq = None  # diffs won't have freq set
    pd.testing.assert_frame_equal(expected, result, check_index_type=True)


def test_id_gaps_no_gaps(ts_timeindex):
    """id_gap_index returns empty Dataframe when threshold exceeds diffs."""
    result = pht.id_gaps(ts_timeindex, pd.Timedelta(hours=4, microseconds=1))
    assert len(result) == 0


def test_category_gaps_frequency(cat_df):
    """Gaps are calculated correctly for categories of varying frequency in Series."""
    duration = pd.Timedelta(days=365)
    delay = pd.Timedelta(days=180)
    gaps = {
        "South Philadelphia": duration - pd.Timedelta(hours=12),
        "San Diego": duration - pd.Timedelta(hours=52),
        "East Midtown": duration - duration / 4,
        "Park South": duration / 2,
        "Quahog": delay,
        "Springfield": pd.Timedelta(hours=0),
    }
    expected = pd.DataFrame(
        gaps.values(), columns=["Cumulative Gap"], index=list(gaps.keys())
    )
    result = pht.category_gaps(cat_df["category"], pd.Timedelta(hours=1))
    pd.testing.assert_frame_equal(expected, result, check_index_type=True)


def test_category_gaps_no_gaps():
    """Series with no gaps should show 0 gaps."""
    start = pd.Timestamp(year=1999, month=1, day=1)
    end = start + pd.Timedelta(hours=1)
    c1 = make_category_data("Springfield", start, end, freq="h")
    c2 = make_category_data("Park South", start, end, freq="2h")
    df = pd.concat([c1, c2])
    gaps = {
        "Springfield": pd.Timedelta(hours=0),
        "Park South": pd.Timedelta(hours=0),
    }
    expected = pd.DataFrame(
        gaps.values(), columns=["Cumulative Gap"], index=list(gaps.keys())
    )
    result = pht.category_gaps(df["category"], pd.Timedelta(hours=1))
    pd.testing.assert_frame_equal(expected, result, check_index_type=True)


def test_category_gaps_nulls():
    """Nulls should be treated as separate categories with correctly calculated gaps."""
    start = pd.Timestamp(year=1999, month=1, day=1)
    end = start + pd.Timedelta(hours=25)  # to get 24 hour range with freq='2h'
    df = make_category_data("Quahog", start, end, freq="2h")
    df.iloc[:2, 3] = None
    df.iloc[2:4, 3] = pd.NA
    df.iloc[4:6, 3] = np.nan
    df.iloc[6:8, 3] = pd.NaT
    gaps = {
        None: pd.Timedelta(hours=23),
        pd.NA: pd.Timedelta(hours=23),
        np.nan: pd.Timedelta(hours=23),
        pd.NaT: pd.Timedelta(hours=23),
        "Quahog": pd.Timedelta(hours=20),
    }
    expected = pd.DataFrame(
        gaps.values(), columns=["Cumulative Gap"], index=list(gaps.keys())
    )
    result = pht.category_gaps(df["category"], pd.Timedelta(hours=1))
    pd.testing.assert_frame_equal(expected, result, check_index_type=True)


def test_category_gaps_not_series_exception(cat_df):
    """Non-series input raises Exception."""
    with pytest.raises(TypeError) as exc:
        pht.category_gaps(cat_df, pd.Timedelta(hours=1))
    assert str(pd.Series) in str(exc.value)


def test_category_gaps_wrong_series_exception():
    """Non-time indexed series raises Exception."""
    series = pd.Series({"A": list(range(5))})
    with pytest.raises(TypeError) as exc:
        pht.category_gaps(series, pd.Timedelta(hours=1))
    assert str(pd.DatetimeIndex) in str(exc.value)


def test_category_gaps_timedelta_wrong_type_exception(cat_df):
    """Wrong input type for threshold raises exception."""
    with pytest.raises(TypeError) as exc:
        pht.category_gaps(cat_df["category"], pd.Timestamp(year=1999, month=1, day=1))
    assert str(pd.Timedelta) in str(exc.value)


def test_category_gaps_warning(cat_df):
    """Series with more categories than max_cat raises warning and returns None."""
    with pytest.warns(UserWarning):
        assert (
            pht.category_gaps(cat_df["category"], pd.Timedelta(hours=1), max_cat=5)
            is None
        )
