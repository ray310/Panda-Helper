"""Panda-Helper is a simple data-profiling utility for Pandas' DataFrames and Series."""

from __future__ import annotations

from pandahelper.profiles import DataFrameProfile, SeriesProfile
from pandahelper.stats import distribution_stats, frequency_table
from pandahelper.times import (
    time_diffs,
    time_diffs_index,
    id_gaps,
    id_gaps_index,
    category_gaps,
)

__version__ = "0.1.1"
__all__ = [
    "frequency_table",
    "distribution_stats",
    "DataFrameProfile",
    "SeriesProfile",
    "time_diffs",
    "time_diffs_index",
    "id_gaps",
    "id_gaps_index",
    "category_gaps",
]
