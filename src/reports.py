""" Classes and associated helper functions to produce simple, pretty-printed
data-profiling reports for Pandas' Series and DataFrames"""

import pandas as pd
import scipy.stats
from tabulate import tabulate


def abbreviate_df(df, first=20, last=5):
    """Returns a shortened series or dataframe containing the first x values
    and last y values. Most useful when input is sorted."""
    if first < 0 or last < 0:
        raise ValueError("'first' and 'last' parameters cannot be negative")
    if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
        raise TypeError(f"{df} is not pd.Series or pd.DataFrame")
    if len(df) <= first + last:
        return df
    else:
        return pd.concat([df.iloc[:first], df.iloc[(len(df) - last): len(df)]])


def abbreviate_string(s, limit=60):
    if not isinstance(s, str):
        raise TypeError("Input is not a string")
    return s[:limit]


def distribution_stats(series):
    """Takes a numeric pd.Series and returns a dictionary of distribution statistics"""
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
    """Takes a pd.Series and returns a frequency table (pd.DataFrame)"""
    freq = series.value_counts()
    freq.name = "Count"
    percent = series.value_counts(normalize=True) * 100
    percent = pd.Series(["{0:.2f}%".format(x) for x in percent], index=percent.index)
    percent.name = "% of Total"
    output = pd.concat([freq, percent], axis=1)
    output.index = [abbreviate_string(str(x), limit=60) for x in output.index]
    return output


class DataFrameReport:
    """Report object created from a pd.DataFrame-like object to pretty-print a simple
    DataFrame-level report. Report can also be saved to a location using the
    save_report method"""

    def __init__(self, df, name=""):
        self.name = name
        self.shape = df.shape
        self.dtypes = [x for x in zip(df.dtypes.index, df.dtypes.values)]
        self.num_duplicates = sum(df.duplicated(keep="first"))
        self.nulls_per_row = df.isna().sum(axis=1)
        self.null_stats = [
            (k, v) for k, v in distribution_stats(self.nulls_per_row).items()
        ]

    def __repr__(self):
        df_info = [
            ("DF Shape", self.shape),
            ("Obviously Duplicated Rows", self.num_duplicates),
        ]
        if self.name:
            df_info.insert(0, ("DF Name", self.name))
        df_table = tabulate(df_info, headers=["DataFrame-Level Info", ""])
        dtype_table = tabulate(self.dtypes, headers=["Column Name", "Data Type"])
        null_table = tabulate(
            self.null_stats, headers=["Summary of Nulls Per Row", ""]
        )
        output = ["".join([x, "\n\n"]) for x in [df_table, dtype_table, null_table]]
        return "".join(output)

    def save_report(self, path):
        with open(path, "w+") as fh:
            fh.write(str(self))


class SeriesReport:
    """Report object created from a pd.Series-like object to pretty-print a simple
    Series-level report. Report can also be saved to a location using the
    save_report method"""

    # attributes
    def __init__(self, series):
        self.name = series.name
        self.dtype = series.dtype
        self.count = series.count()  # counts non-null values
        self.num_unique = series.nunique()
        self.num_nulls = series.size - self.count  # NAs, NANs, but not " "
        self.frequency = frequency_table(series)
        self.stats = None
        if pd.api.types.is_numeric_dtype(self.dtype):
            self.stats = [(k, v) for k, v in distribution_stats(series).items()]

    def __repr__(self):
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
        with open(path, "w+") as fh:
            fh.write(str(self))
