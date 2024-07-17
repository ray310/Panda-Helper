"""Panda-Helper time-series functions."""

from typing import Union  # TODO: Remove when deprecating Python 3.9
import pandas as pd
import pandas.api.types as pat


def time_diffs(series: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series(pd.Timedelta):
    """Calculate time difference between subsequent observations.

    Sorts input by time before calculating diffs.

    Args:
        series (pd.Series or pd.DatetimeIndex): Pandas Series or DatetimeIndex
            to calculate time diffs on.

    Returns:
        Series of diffs (gaps) indexed by the time the diff was calculated.

    Raises:
        TypeError: If input is not Series of type datetime64 or DatetimeIndex.

    Examples:
        Calculate time differences between observations on Series of timestamps after
        it has been randomized:

        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> start = pd.Timestamp(year=1999, month=1, day=1)
        >>> rng = pd.date_range(start, periods=10, freq="D").delete([3, 4, 5, 8])
        >>> series = pd.Series(rng).sample(frac=1, random_state=3)  # randomize order

        >>> ph.time_diffs(series)
        1999-01-01      NaT
        1999-01-02   1 days
        1999-01-03   1 days
        1999-01-07   4 days
        1999-01-08   1 days
        1999-01-10   2 days
        Name: diffs, dtype: timedelta64[ns]
    """
    if not pat.is_datetime64_any_dtype(series.dtype):
        raise TypeError("Should be of datetime64 dtype.")
    series = series.sort_values()
    diffs = pd.Series(series.diff(), name="diffs")
    diffs.index = series
    return diffs


def time_diffs_index(df: Union[pd.Series, pd.DataFrame]) -> pd.Series(pd.Timedelta):
    """Calculate time difference between subsequent time-indexed observations.

    Sorts input by time index before calculating diffs.

    Args:
        df (pd.Series or pd.DataFrame): Pandas Series or DataFrame with DateTimeIndex
            to calculate time diffs on.

    Returns:
        Series of diffs (gaps) indexed by the time the diff was calculated.

    Raises:
        TypeError: If input does not have a DatetimeIndex.

    Examples:
        Calculate time differences between observations on time-indexed DataFrame after
        it has been randomized:

        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> start = pd.Timestamp(year=1999, month=1, day=1)
        >>> rng = pd.date_range(start, periods=10, freq="D").delete([3, 4, 5, 8])
        >>> # index by time then randomize order
        >>> df = pd.DataFrame(range(len(rng)), index=rng).sample(frac=1, random_state=3)

        >>> ph.time_diffs_index(df)
        1999-01-01      NaT
        1999-01-02   1 days
        1999-01-03   1 days
        1999-01-07   4 days
        1999-01-08   1 days
        1999-01-10   2 days
        Name: diffs, dtype: timedelta64[ns]
    """
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        diffs = pd.Series(df.index.diff(), name="diffs")
        diffs.index = df.index
        return diffs
    raise TypeError(f"Index should be of type {pd.DatetimeIndex}")


def id_gaps(
    series: Union[pd.Series, pd.DatetimeIndex], threshold: pd.Timedelta
) -> pd.DataFrame:
    """Identify time gaps above `threshold` in datetime64 Series or DatetimeIndex.

    Sorts input by time before calculating gaps.

    Args:
        series (pd.Series or pd.DatetimeIndex): `datetime64` Series or DatetimeIndex.
        threshold (pd.Timedelta): Threshold to identify gaps
            (and not expected time differences).

    Returns:
        One-column Pandas DataFrame of gaps indexed by when gap was calculated.

    Examples:
        Identify time gaps on Series of timestamps with a 2 and 4 hour
        gap after it has been randomized:

        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> start = pd.Timestamp(year=1999, month=1, day=1)
        >>> rng = pd.date_range(start, periods=24, freq="1h").delete([3, 4, 8, 9, 10])
        >>> series = pd.Series(rng).sample(frac=1, random_state=3)  # randomize order

        >>> ph.id_gaps(series, pd.Timedelta(hours=1))
                                      diffs
        1999-01-01 11:00:00 0 days 04:00:00
        1999-01-01 04:00:00 0 days 02:00:00
    """
    diffs = time_diffs(series)
    return diffs[diffs > threshold].sort_values(ascending=False).to_frame()


def id_gaps_index(
    df: Union[pd.Series, pd.DataFrame], threshold: pd.Timedelta
) -> pd.DataFrame:
    """Identify time gaps above `threshold` in time-indexed Series or DataFrame.

    Sorts input by time index before calculating diffs.

    Args:
        df (pd.Series or pd.DataFrame): Time-indexed Series or DataFrame.
        threshold (pd.Timedelta): Threshold to identify gaps
            (and not expected time differences).

    Returns:
        One-column Pandas DataFrame of gaps indexed by when gap was calculated.

    Examples:
        Identify time gaps on an hourly, time-indexed Series with a 2 and 4 hour
        gap after it has been randomized:

        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> start = pd.Timestamp(year=1999, month=1, day=1)
        >>> rng = pd.date_range(start, periods=24, freq="1h").delete([3, 8, 9, 10])
        >>> # index by time then randomize order
        >>> df = pd.DataFrame(range(len(rng)), index=rng).sample(frac=1, random_state=3)

        >>> ph.id_gaps_index(df, pd.Timedelta(hours=1))
                                      diffs
        1999-01-01 11:00:00 0 days 04:00:00
        1999-01-01 04:00:00 0 days 02:00:00
    """
    diffs = time_diffs_index(df)
    return diffs[diffs > threshold].sort_values(ascending=False).to_frame()
