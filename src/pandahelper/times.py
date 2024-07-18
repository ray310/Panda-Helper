"""Panda-Helper time-series functions."""

from warnings import warn
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


def category_gaps(
    series: pd.Series, threshold: pd.Timedelta, max_cat: int = 50
) -> [pd.DataFrame, None]:
    """Calculate sum of gaps for each category in time-indexed Series.

    Gaps are time differences in excess of expected time increment (threshold). Gap per
    category is relative to the minimum and maximum times in the Series.
    Intended for use with categorical-like Series.

    Args:
        series (pd.Series): Categorical-like Series.
        threshold (pd.Timedelta): Threshold for the time difference to be considered
            a gap. For hourly data, threshold should be pd.Timedelta(hours=1).
        max_cat (int): Maximum number categories (unique values) before issuing
            warning and returning `None`.

    Returns:
        Key-value pairs with category name and associated gap. Will return None if
            number of categories exceeds `max_cat`.

    Warns:
        UserWarning: If the number of categories (unique values) in the series
            exceeds `max_cat`.

    Examples:
        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> start = pd.Timestamp(year=1999, month=1, day=1)
        >>> a = pd.Series(["A"] * 30, index=pd.date_range(start, periods=30, freq="D"))
        >>> b = pd.Series(["B"] * 15, index=pd.date_range(start, periods=15, freq="2D"))
        >>> c = pd.Series(["C"] * 10, index=pd.date_range(start, periods=10, freq="D"))
        >>> ph.category_gaps(pd.concat([a, b, c]), threshold=pd.Timedelta(days=1))
                      Cumulative Gap
            C        20 days
            B        15 days
            A         0 days
    """
    if not isinstance(series, pd.Series) or not isinstance(
        series.index, pd.DatetimeIndex
    ):
        raise TypeError(
            f"Series should be {pd.Series} with index of type {pd.DatetimeIndex}"
        )
    if not isinstance(threshold, pd.Timedelta):
        raise TypeError(f"Increment should be {pd.Timedelta}")
    gaps = {}
    time_range = series.index.max() - series.index.min()
    categories = series.unique()
    if len(categories) > max_cat:
        msg = (
            f"Number of categories is greater than f{max_cat}. To proceed "
            f"increase 'max_cat' and run function again."
        )
        warn(msg, stacklevel=2)
        return None
    for cat in categories:
        cat_slice = series.loc[series == cat]
        if pd.isnull(cat):  # treat nulls as distinct category
            nulls = series.apply(lambda x: x is cat)  # pylint: disable=W0640
            cat_slice = series[nulls]
        cat_range = cat_slice.index.max() - cat_slice.index.min()
        diffs = time_diffs_index(cat_slice)
        gap = (diffs[diffs > threshold] - threshold).sum()
        gaps[cat] = time_range - cat_range + gap
    df = pd.Series(gaps.values(), index=gaps.keys(), name="Cumulative Gap")
    return df.sort_values(ascending=False).to_frame()
