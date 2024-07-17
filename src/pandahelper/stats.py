"""Panda-Helper statistics functions."""

import numpy as np
import pandas as pd
import pandas.api.types as pat
import scipy.stats


def frequency_table(series: pd.Series) -> pd.DataFrame:
    """Return value counts and relative frequency.

    Args:
        series (pd.Series): Pandas Series used to calculate value counts and relative
            frequencies.

    Returns:
        Pandas DataFrame of value counts and percentages indexed by value.

    Raises:
        TypeError: If input is not a Pandas Series.

    Examples:
        >>> import random
        >>> import pandahelper as ph
        >>>
        >>> random.seed(314)
        >>> cities = ["Springfield", "Quahog", "Philadelphia", "Shelbyville"]
        >>> series = pd.Series(random.choices(cities, k = 200))
        >>> ph.frequency_table(series)
                          Count % of Total
            Springfield      66     33.00%
            Quahog           51     25.50%
            Philadelphia     44     22.00%
            Shelbyville      39     19.50%
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


def _abbreviate_string(s, limit=60) -> str:
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
    """Return single-column Pandas DataFrame of distribution statistics.

    Args:
        series (pd.Series): Pandas Series used to calculate distribution statistics.
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

    Examples:
        Distribution stats for Pandas Series of type `float64`:
        >>> from random import seed, gauss, expovariate
        >>> import pandahelper as ph
        >>> import pandas as pd
        >>>
        >>> seed(314)
        >>> series = pd.Series([gauss(mu=30, sigma=20) for x in range(200)])
        >>> ph.distribution_stats(series)
                                       Statistic Value
            count                           200.000000
            min                             -23.643007
            1%                              -11.918955
            5%                                2.833604
            25%                              17.553793
            50%                              31.420759
            75%                              42.074998
            95%                              60.305435
            99%                              72.028633
            max                              81.547828
            mean                             30.580535
            standard deviation               18.277706
            median                           31.420759
            median absolute deviation        12.216607
            skew                             -0.020083

        Distribution stats for Pandas Series of type `datetime64`:
        >>> start = pd.Timestamp(2000, 1, 1)
        >>> tds = [pd.Timedelta(hours=int(expovariate(lambd=.003))) for x in range(200)]
        >>> times = [start + td for td in tds]
        >>> series = pd.Series(times)
        >>> ph.distribution_stats(series)
                                       Statistic Value
        count                                      200
        min                        2000-01-01 00:00:00
        1%                         2000-01-01 01:59:24
        5%                         2000-01-01 09:00:00
        25%                        2000-01-04 08:00:00
        50%                        2000-01-08 04:30:00
        75%                        2000-01-16 21:00:00
        95%                        2000-02-08 01:36:00
        99%                        2000-02-22 10:20:24
        max                        2000-04-01 17:00:00
        mean                       2000-01-12 14:24:18
        standard deviation  12 days 16:47:15.284423042
        median                     2000-01-08 04:30:00
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

    Warning:
        This function is subject to change without warning.

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


def _order_stats(stats: dict) -> dict:
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
