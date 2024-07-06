"""Panda-Helper Classes and associated helper functions."""

from warnings import warn
import numpy as np
import pandas as pd
import pandas.api.types as pat
import scipy.stats
from tabulate import tabulate


warn(
    "reports module is deprecated and will be removed in a future version."
    "'import pandahelper' will provide access to profiles and methods.",
    DeprecationWarning,
    stacklevel=2,
)


def _abbreviate_df(df, first=20, last=5):
    """Return a shortened DataFrame or Series.

    Returned dataframe will contain the input numbers of the first and last
    rows. Most useful when input is sorted.

    Args:
        df (pd.DataFrame or pd.Series): pd.DataFrame or pd.Series
        first (int): First n rows of input to be included in the shortened
            Dataframe or Series. Default value is 20.
        last (int): Last n rows of input to be included in the shortened
            Dataframe or Series. Default value is 5.

    Returns:
        pd.DataFrame or pd.Series: Shortened DataFrame or Series.

    Raises:
        ValueError: If first or last are negative.
        TypeError: If df is not a pd.DataFrame or pd.Series.
    """
    if first < 0 or last < 0:
        raise ValueError("'first' and 'last' parameters cannot be negative")
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{df} is not pd.Series or pd.DataFrame")
    if len(df) <= first + last:
        abbrev = df
    else:
        abbrev = pd.concat([df.iloc[:first], df.iloc[(len(df) - last) : len(df)]])
    return abbrev


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


def distribution_stats(series):
    """Return Series distribution statistics.

    Args:
        series (pd.Series): Series used to calculate distribution statistics.

    Returns:
        dict: Key-value pairs with name of statistic and calculated value.

    Raises:
        TypeError: If input is not a numeric pd.Series.
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
            f"distribution_stats is not supported for pd.Series of type: {series.dtype}"
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


def frequency_table(series):
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
    return output


class DataFrameProfile:
    """DataFrame-level data profile.

    Prepares pretty-printed DataFrame-level data profile that can be displayed
    or saved.

    Attributes:
        name (str): Name of DataFrame profile if provided. Default value is "".
        shape (tuple): Dataframe shape.
        dtypes (pd.Series): Data types of Series within DataFrame.
        num_duplicates (int): Number of duplicated rows.
        nulls_per_row (pd.Series): Count of null values per row.
        nulls_stats (list): Distribution statistics on nulls per row.
    """

    def __init__(self, df, name=""):
        """Initialize DataFrameProfile.

        Args:
            df (pd.DataFrame): DataFrame to profile.
            name (str, optional): Name to assign to profile.

        Raises:
            TypeError: If input is not a pd.DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df}, is not pd.DataFrame")
        self.name = name
        self.shape = df.shape
        self.dtypes = list(zip(df.dtypes.index, df.dtypes.values))
        self.num_duplicates = sum(df.duplicated(keep="first"))
        self.nulls_per_row = df.isna().sum(axis=1)
        self.null_stats = distribution_stats(self.nulls_per_row)

    def __repr__(self):
        """Printable version of profile."""
        df_info = [
            ("DF Shape", self.shape),
            ("Duplicated Rows", self.num_duplicates),
        ]
        if self.name:
            df_info.insert(0, ("DF Name", self.name))
        df_table = tabulate(df_info, headers=["DataFrame-Level Info", ""])
        dtype_table = tabulate(self.dtypes, headers=["Series Name", "Data Type"])
        null_table = tabulate(
            list(self.null_stats.items()), headers=["Summary of Nulls Per Row", ""]
        )
        output = ["".join([x, "\n\n"]) for x in [df_table, dtype_table, null_table]]
        return "".join(output)

    def save_report(self, path):
        """Save profile to provided path.

        Args:
            path (str): Where to save profile.
        """
        with open(path, "w+", encoding="utf-8") as fh:
            fh.write(str(self))


class SeriesProfile:
    """Series-level data profile.

    Prepares pretty-printed Series-level data profile that can be displayed
    or saved.

    Attributes:
        name (str): Name of Series.
        dtype (np.dtype): Data types of Series within DataFrame.
        count (int): Count of non-null values.
        num_unique (int): Number of unique values.
        num_nulls (int): Number of null values.
        frequency (pd.DataFrame): Table of value counts and relative frequency
            as a DataFrame
        stats (list): Distribution statistics for numeric Series.
    """

    def __init__(self, series):
        """Initialize SeriesProfile.

        Args:
            series (pd.Series): DataFrame to profile.

        Raises:
            TypeError: If input is not a pd.Series.
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"{series}, is not pd.DataFrame")
        self.name = series.name
        self.dtype = series.dtype
        self.count = series.count()  # counts non-null values
        self.num_unique = series.nunique()
        self.num_nulls = series.size - self.count  # NAs, nans, NaT, but not ""
        self.frequency = frequency_table(series)
        self.stats = None
        if not (
            pat.is_object_dtype(self.dtype)
            or isinstance(self.dtype, pd.CategoricalDtype)
        ):
            self.stats = distribution_stats(series)

    def __repr__(self):
        """Printable version of profile."""
        series_info = [
            ("Data Type", self.dtype),
            ("Count", self.count),
            ("Unique Values", self.num_unique),
            ("Null Values", self.num_nulls),
        ]
        sname = self.name
        if not sname:
            sname = "Series"
        series_table = tabulate(series_info, headers=[f"{sname} Info", ""])
        freq_info = _abbreviate_df(self.frequency, first=20, last=5)
        freq_table = tabulate(freq_info, headers=["Value", "Count", "% of total"])
        stats_table = ""
        if self.stats is not None:
            stats = self.stats
            if pat.is_complex_dtype(
                self.dtype
            ):  # tabulate converts complex numbers to real numbers
                stats = {k: str(v) for k, v in self.stats.items()}
            stats_table = tabulate(list(stats.items()), headers=["Statistic", "Value"])
        output = ["".join([x, "\n\n"]) for x in [series_table, freq_table, stats_table]]
        return "".join(output)

    def save_report(self, path):
        """Save profile to provided path.

        Args:
            path (str): Where to save profile.
        """
        with open(path, "w+", encoding="utf-8") as fh:
            fh.write(str(self))
