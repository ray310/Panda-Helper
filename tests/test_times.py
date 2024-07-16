"""Tests for functions in times.py."""

import pandas as pd
import pytest
import pandahelper.times as pht


def test_time_diffs_index(cat_df):
    """time_diffs_index should work on shuffled pd.Series or pd.DataFrame."""
    # test DF
    df_result = pht.time_diffs_index(cat_df)
    assert df_result.iloc[0] is pd.NaT
    assert all(df_result[1:] == pd.Timedelta(hours=1))
    # test Series
    series_result = pht.time_diffs_index(cat_df["B"])
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


def test_time_diffs(cat_df):
    """time_diffs should work on shuffled pd.Series or Index of timestamps."""
    valid = [cat_df.index, pd.Series(cat_df.index)]
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
