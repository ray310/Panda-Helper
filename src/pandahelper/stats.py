"""Panda-Helper statistics functions."""

import numpy as np
import pandas as pd
import pandas.api.types as pat
import scipy.stats


def frequency_table(series: pd.Series) -> pd.DataFrame:
    """Return value counts and relative frequency.

    Args:
        series (pd.Series): Series used to calculate value counts and relative
            frequencies.

    Returns:
        pd.DataFrame: DataFrame containing values as the row index with value
            counts and counts as a percentage of total count.

    Raises:
        TypeError: If input is not a pd.Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{series}, is not pd.Series")
    freq = series.value_counts()  # excludes nulls
    freq.name = "Count"
    counts = series.value_counts(normalize=True)
    percent = pd.Series([f"{x:.2%}" for x in counts], index=counts.index)
    percent.name = "% of Total"
    output = pd.concat([freq, percent], axis=1)
    output.index = [_abbreviate_string(str(x), limit=60) for x in output.index]
    return output.sort_values(by="Count", ascending=False)


def _abbreviate_string(s, limit=60):
    """Return first x characters of a string.

    Args:
        s (str): String to be shortened
        limit (int): Maximum length of string. Default value is 60.

    Returns:
        str: Shortened string.

    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input is not a string")
    return s[:limit]


def distribution_stats(series: pd.Series) -> pd.DataFrame:
    """Return single-column pd.DataFrame of distribution statistics for pd.Series.

    Args:
        series (pd.Series): Series used to calculate distribution statistics.
        Distribution statistics will depend on series dtype. Supported dtypes are:
            - int64
            - float64
            - bool
            - complex128
            - datetime64
            - timedelta64
            - period[<unit>]
            - interval

    Returns:
        pd.DataFrame: Single-column of calculated values with statistics as index.

    Raises:
        TypeError: If input is not a numeric-like pd.Series.
    """
    stats = dist_stats_dict(series)
    return pd.DataFrame.from_dict(stats, orient="index", columns=["Statistic Value"])


def dist_stats_dict(series: pd.Series) -> dict:
    """Return dictionary of distribution statistics for pd.Series.

    Args:
        series (pd.Series): Series used to calculate distribution statistics.
        Distribution statistics will depend on series dtype. Supported dtypes are:
            - int64
            - float64
            - bool
            - complex128
            - datetime64
            - timedelta64
            - period[<unit>]
            - interval

    Returns:
        dict: Key-value pairs with name of statistic and calculated value.

    Raises:
        TypeError: If input is not a numeric-like pd.Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{series}, is not pd.Series")
    if not (
        pat.is_numeric_dtype(series.dtype)
        or pat.is_datetime64_any_dtype(series.dtype)
        or pat.is_timedelta64_dtype(series.dtype)
        or isinstance(series.dtype, (pd.PeriodDtype, pd.IntervalDtype))
    ):
        raise TypeError(
            f"Distribution Stats not supported for pd.Series of type: {series.dtype}"
        )
    stats = {
        "count": series.count(),
        "min": series.min(),
        "max": series.max(),
    }
    if isinstance(series.dtype, pd.IntervalDtype):
        return _order_stats(stats)
    if isinstance(series.dtype, pd.PeriodDtype):
        stats["median"] = series.median()
        _add_quantiles(series, stats)
        return _order_stats(stats)

    stats["mean"] = series.mean()
    if pd.api.types.is_bool_dtype(series.dtype):
        return _order_stats(stats)

    if pd.api.types.is_complex_dtype(series.dtype):
        stats["median"] = np.median(series)  # pd.median struggles here
        return _order_stats(stats)

    stats["median"] = series.median()
    _add_quantiles(series, stats)
    stats["standard deviation"] = series.std()
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return _order_stats(stats)

    mad = scipy.stats.median_abs_deviation(series, nan_policy="omit")
    stats["median absolute deviation"] = mad
    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return _order_stats(stats)

    stats["skew"] = series.skew()
    return _order_stats(stats)  # int / float


def _add_quantiles(series: pd.Series, d: dict):
    """Add quantiles to distribution_stats."""
    d["1%"] = series.quantile(0.01)
    d["5%"] = series.quantile(0.05)
    d["25%"] = series.quantile(0.25)
    d["50%"] = series.quantile(0.50)
    d["75%"] = series.quantile(0.75)
    d["95%"] = series.quantile(0.95)
    d["99%"] = series.quantile(0.99)


def _order_stats(stats: dict):
    """Sort stats dictionary by order provided in all_stats.

    Helper function used in distribution_stats.
    """
    all_stats = [
        "count",
        "min",
        "1%",
        "5%",
        "25%",
        "50%",
        "75%",
        "95%",
        "99%",
        "max",
        "mean",
        "standard deviation",
        "median",
        "median absolute deviation",
        "skew",
    ]
    stats_order = dict(zip(all_stats, range(len(all_stats))))
    key_list = sorted(stats.keys(), key=lambda k: stats_order[k])
    return {k: stats[k] for k in key_list}
