"""Panda-Helper Classes and associated helper functions."""

import pandas as pd
import scipy.stats
from tabulate import tabulate


def abbreviate_df(df, first=20, last=5):
    """Returns a shortened DataFrame or Series.

    Returned dataframe will contain the input numbers of the first and last
    rows. Most useful when input is sorted.

    Args:
        df (pd.DataFrame or pd.Series): pd.DataFrame or pd.Series
        first (int): First n rows of input to be included in the shortened
            Dataframe or Series. Default value is 20.
        first (int): Last n rows of input to be included in the shortened
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


def abbreviate_string(s, limit=60):
    """Returns first x characters of a string.

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
    """Returns Series distribution statistics.

    Args:
        series (pd.Series): Series used to calculate distribution statistics.

    Returns:
        dict: Key-value pairs with name of statistic and calculated value.

    Raises:
        TypeError: If input is not a numeric pd.Series.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"{series}, is not pd.Series")
    if not pd.api.types.is_numeric_dtype(series.dtype):
        raise TypeError(f"{series}, is not numeric")
    mad = scipy.stats.median_abs_deviation(series, nan_policy="omit")
    stats = {
        "count": series.count(),
        "min": series.min(),
        "1%": series.quantile(0.01),
        "5%": series.quantile(0.05),
        "25%": series.quantile(0.25),
        "50%": series.quantile(0.50),
        "75%": series.quantile(0.75),
        "95%": series.quantile(0.95),
        "99%": series.quantile(0.99),
        "max": series.max(),
        "median": series.median(),
        "mean": series.mean(),
        "median absolute deviation": mad,
        "standard deviation": series.std(),
        "skew": series.skew(),
    }
    return stats


def frequency_table(series):
    """Returns value counts and relative frequency.

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
    output.index = [abbreviate_string(str(x), limit=60) for x in output.index]
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
        """Initializes DataFrameProfile.

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
        self.null_stats = list(distribution_stats(self.nulls_per_row).items())

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
        null_table = tabulate(self.null_stats, headers=["Summary of Nulls Per Row", ""])
        output = ["".join([x, "\n\n"]) for x in [df_table, dtype_table, null_table]]
        return "".join(output)

    def save_report(self, path):
        """Saves profile to provided path.

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
        """Initializes SeriesProfile.

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
        self.num_nulls = series.size - self.count  # NAs, NANs, but not " "
        self.frequency = frequency_table(series)
        self.stats = None
        if pd.api.types.is_numeric_dtype(self.dtype):
            self.stats = list(distribution_stats(series).items())

    def __repr__(self):
        """Printable version of profile."""
        series_info = [
            ("Data Type", self.dtype),
            ("Count", self.count),
            ("Unique Values", self.num_unique),
            ("Null Values", self.num_nulls),
        ]
        series_table = tabulate(series_info, headers=[f"{self.name} Info", ""])
        freq_info = abbreviate_df(self.frequency, first=20, last=5)
        freq_table = tabulate(freq_info, headers=["Value", "Count", "% of total"])
        stats_table = ""
        if self.stats is not None:
            stats_table = tabulate(self.stats, headers=["Statistic", "Value"])

        output = ["".join([x, "\n\n"]) for x in [series_table, freq_table, stats_table]]
        return "".join(output)

    def save_report(self, path):
        """Saves profile to provided path.

        Args:
            path (str): Where to save profile.
        """
        with open(path, "w+", encoding="utf-8") as fh:
            fh.write(str(self))
