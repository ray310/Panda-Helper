"""Panda-Helper time-series functions."""

import pandas as pd
import pandas.api.types as pat


def time_diffs(series: pd.Series | pd.DatetimeIndex) -> pd.Series(pd.Timedelta):
    """Calculate time diffs (gaps) for Pandas Series or Index of timestamps.

    Sorts input by time before calculating diffs.

    Args:
        series (pd.Series or pd.DatetimeIndex): Pandas Series or DatetimeIndex
            to calculate time diffs on.

    Returns:
        Series of diffs (gaps) indexed by the time the diff was calculated.

    Raises:
        TypeError: If input is not Series of type datetime64 or DatetimeIndex.
    """
    if not pat.is_datetime64_any_dtype(series.dtype):
        raise TypeError("Should be Series of datetime64 dtype.")
    series = series.sort_values()
    diffs = pd.Series(series.diff(), name="diffs")
    diffs.index = series
    return diffs


def time_diffs_index(df: pd.DataFrame | pd.Series) -> pd.Series(pd.Timedelta):
    """Calculate time diffs (gaps) for time-indexed Pandas Series or Dataframe.

    Sorts input by time before calculating diffs.

    Args:
        df (pd.Series or pd.DataFrame): Pandas Series or DataFrame with DateTimeIndex
            to calculate time diffs on.

    Returns:
        Series of diffs (gaps) indexed by the time the diff was calculated.

    Raises:
        TypeError: If input does not have a DatetimeIndex.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        diffs = pd.Series(df.index.diff(), name="diffs")
        diffs.index = df.index
        return diffs
    raise TypeError(f"Index should be of type {pd.DatetimeIndex}")
