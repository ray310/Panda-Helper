"""Panda-Helper data profiles."""

import bs4
import pandas as pd
import pandas.api.types as pat
from tabulate import tabulate
import pandahelper.stats as phs
import pandahelper.times as pht


class DataFrameProfile:
    """Pandas DataFrame data profile.

    Prepare data profile of Pandas DataFrame that can be displayed
    or saved.

    Attributes:
        name (str): Name of DataFrame profile if provided. Default value is "".
        shape (tuple): Dataframe shape.
        dtypes (pd.Series): Data types of DataFrame index and Series in DataFrame.
        memory_usage (pd.Series): Memory usage (MB) of index and Series in DataFrame.
        num_duplicates (int): Number of duplicated rows.
        nulls_per_row (pd.Series): Count of null values per row.
        null_stats (list): Distribution statistics on nulls per row.
        time_diffs (pd.Series): Time diffs (gaps) if DataFrame has a DateTimeIndex.
    """

    def __init__(self, df: pd.DataFrame, *, name: str = "", fmt: str = "simple"):
        """Initialize DataFrameProfile.

        Args:
            df (pd.DataFrame): DataFrame to profile.
            name (str: optional): Name to assign to profile.
            fmt (str: optional): Printed table format. See
                https://github.com/astanin/python-tabulate for options.

        Raises:
            TypeError: If input is not a Pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{df}, is not pd.DataFrame")
        self.name = name
        self.shape = df.shape
        self.dtypes = pd.concat([pd.Series(df.index.dtype, index=["Index"]), df.dtypes])
        self.memory_usage = df.memory_usage(index=True, deep=True) / 1000000  # MB
        self.num_duplicates = sum(df.duplicated(keep="first"))
        self.nulls_per_row = df.isna().sum(axis=1)
        self.time_diffs = self.__calc_time_diffs(df)
        self.null_stats = self.__null_stats()
        self._format = fmt

    def __null_stats(self, delete_key="count"):
        """Prepare distribution statistics for the number of nulls per row."""
        stats = phs.dist_stats_dict(self.nulls_per_row)
        new_stats = {"Number of Columns": self.shape[1]}
        del stats[delete_key]
        return new_stats | stats

    @staticmethod
    def __calc_time_diffs(df: pd.DataFrame) -> pd.Series or None:
        """Calculate time diffs if DataFrame is time-indexed."""
        if pat.is_datetime64_any_dtype(df.index):
            return pht.time_diffs_index(df)
        return None

    def __create_tables(self, table_fmt: str):
        """Create DataFrameProfile summary tables.

        Args:
            table_fmt (str): Tabulate table format name.

        Returns:
            list(str): List of Tabulate tables.
        """
        df_info = [
            ("DF Shape", self.shape),
            ("Duplicated Rows", self.num_duplicates),
            ("Memory Usage (MB)", f"{self.memory_usage.sum():,.3f}"),
        ]
        if self.name:
            df_info.insert(0, ("DF Name", self.name))
        df_table = tabulate(
            df_info, headers=["DataFrame-Level Info", ""], tablefmt=table_fmt
        )
        type_usage = pd.concat(
            [self.dtypes, self.memory_usage], keys=["dtype", "memory"], axis=1
        )  # match on index
        dtype_usage_table = tabulate(
            list(
                zip(
                    type_usage.index,
                    type_usage["dtype"].values,
                    type_usage["memory"].values,
                )
            ),
            headers=["Series Name", "Data Type", "Memory Usage (MB)"],
            tablefmt=table_fmt,
        )
        null_table = tabulate(
            list(self.null_stats.items()),
            headers=["Summary of Nulls Per Row", ""],
            tablefmt=table_fmt,
        )
        tables = [df_table, dtype_usage_table, null_table]
        if self.time_diffs is not None:
            time_diffs_table = tabulate(
                phs.frequency_table(self.time_diffs),
                headers=["Time Diffs", "Count", "% of total"],
                tablefmt=table_fmt,
            )
            tables.append(time_diffs_table)
        return tables

    def __repr__(self):
        """Printable version of profile."""
        output = ["".join([x, "\n\n"]) for x in self.__create_tables(self._format)]
        return "".join(output).strip() + "\n"

    def _repr_html_(self):
        """HTML representation of profile."""
        tables = [_format_html_table(t) for t in self.__create_tables("html")]
        tables[1] = _decimal_align_col(tables[1], 2)  # type/memory usage table
        tables[2] = _decimal_align_col(tables[2], 1)  # stats table
        output = "".join([table + "<br>" for table in tables])
        return output[:-4]  # remove last <br>

    def save(self, path: str):
        """Save profile to provided path.

        Args:
            path (str): Where to save profile.
        """
        with open(path, "w+", encoding="utf-8") as fh:
            fh.write(str(self))


class SeriesProfile:
    """Pandas Series data profile.

    Prepare data profile of Pandas Series that can be displayed
    or saved.

    Attributes:
        name (str): Name of Series.
        dtype (numpy.dtype or Pandas dtype): Data types of Series within DataFrame.
        count (int): Count of non-null values.
        num_unique (int): Number of unique values.
        num_nulls (int): Number of null values.
        frequency (pd.DataFrame): Frequency table with counts and percentage.
        stats (dict): Distribution statistics for Series.
        time_diffs (pd.Series): Time diffs (gaps) if series is of type `datetime64`.
            Alternately, can be time diffs in a Series with a DateTimeIndex if the
            `time_index` parameter was set to `True` when creating Series Profile.
    """

    def __init__(
        self,
        series: pd.Series,
        *,
        fmt: str = "simple",
        freq_most_least: tuple = (10, 5),
        time_index: bool = False,
    ):
        """Initialize SeriesProfile.

        Args:
            series (pd.Series): Pandas Series to profile.
            fmt (str: optional): Printed table format. See:
                <https://github.com/astanin/python-tabulate> for options.
            freq_most_least (tuple: optional): Tuple (x, y) of the x most common and
                y least common values to display in frequency table.
            time_index (bool: optional): Whether to use the index for calculating time
                diffs for a `datetime64`-related Pandas Series. Not relevant for
                non-time related Series.

        Raises:
            TypeError: If input is not a Pandas Series.
        """
        if not isinstance(series, pd.Series):
            raise TypeError(f"{series}, is not pd.Series")
        if freq_most_least[0] < 0 or freq_most_least[1] < 0:
            raise ValueError("Tuple values must be >= 0!")
        self._format = fmt
        self._freq_table = freq_most_least
        self.name = series.name
        self.dtype = series.dtype
        self.count = series.count()  # counts non-null values
        self.num_unique = series.nunique()
        self.num_nulls = series.size - self.count  # NAs, nans, NaT, but not ""
        self.frequency = phs.frequency_table(series)
        self.stats = self.__calc_stats(series)
        self.time_diffs = self.__calc_time_diffs(series, time_index)

    def __calc_stats(self, series: pd.Series):
        """Calculate distribution stats if allowed dtype, else return None."""
        if pat.is_object_dtype(self.dtype) or isinstance(
            self.dtype, pd.CategoricalDtype
        ):
            return None
        return phs.dist_stats_dict(series)

    @staticmethod
    def __calc_time_diffs(series: pd.Series, use_time_index: bool) -> pd.Series or None:
        """Calculate time diffs for time-indexed series or datetime64 series."""
        if use_time_index and pat.is_datetime64_any_dtype(series.index):
            return pht.time_diffs_index(series)
        if (not use_time_index) and pat.is_datetime64_any_dtype(series):
            return pht.time_diffs(series)
        return None

    def __create_tables(self, table_fmt: str) -> list[str]:
        """Create and return SeriesProfile summary tables."""
        series_info = [
            ("Data Type", self.dtype),
            ("Count", self.count),
            ("Unique Values", self.num_unique),
            ("Null Values", self.num_nulls),
        ]
        sname = self.name
        if not sname:
            sname = "Series"
        series_table = tabulate(
            series_info, headers=[f"{sname} Info", ""], tablefmt=table_fmt
        )
        freq_info = _abbreviate_df(
            self.frequency, first=self._freq_table[0], last=self._freq_table[1]
        )
        freq_table = tabulate(
            freq_info, headers=["Value", "Count", "% of total"], tablefmt=table_fmt
        )
        tables = [series_table, freq_table]
        if self.stats is not None:
            stats = self.stats
            # tabulate casts complex numbers to real numbers, dropping imaginary part
            if pat.is_complex_dtype(self.dtype):
                stats = {k: str(v) for k, v in self.stats.items()}
            stats_table = tabulate(
                list(stats.items()),
                headers=["Statistic", "Value"],
                tablefmt=table_fmt,
            )
            tables.append(stats_table)
        if self.time_diffs is not None:
            time_diffs_table = tabulate(
                phs.frequency_table(self.time_diffs),
                headers=["Time Diffs", "Count", "% of total"],
                tablefmt=table_fmt,
            )
            tables.append(time_diffs_table)
        return tables

    def __repr__(self):
        """Printable version of profile."""
        output = ["".join([x, "\n\n"]) for x in self.__create_tables(self._format)]
        return "".join(output).strip() + "\n"

    def _repr_html_(self):
        """HTML representation of profile."""
        tables = [_format_html_table(t) for t in self.__create_tables("html")]
        if self.stats is not None:
            tables[2] = _decimal_align_col(tables[2], 1)
        output = "".join([table + "<br>" for table in tables])
        return output[:-4]  # remove last <br>

    def save(self, path):
        """Save profile to provided path.

        Args:
            path (str): Where to save profile.
        """
        with open(path, "w+", encoding="utf-8") as fh:
            fh.write(str(self))


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


def _format_html_table(table: str, align: str = "left", font: str = "monospace") -> str:
    """Add additional formatting to HTML table prepared by tabulate."""
    soup = bs4.BeautifulSoup(table, "html.parser")
    for row in soup.find_all("tr"):
        tags = row.find_all(["th", "td"])  # row in thead will have 'th'
        for tag in tags:
            tag["style"] = f"font-family: {font}, monospace; text-align: {align};"
    return str(soup)


def _decimal_align_col(table: str, col: int):
    """Create decimal-aligned numbers in column of HTML table."""
    soup = bs4.BeautifulSoup(table, "html.parser")
    for row in soup.find_all("tr"):
        tags = row.find_all("td")
        if tags:
            tags[col].string = tags[col].string.replace(" ", "\u2007")  # figure space
    return str(soup)
