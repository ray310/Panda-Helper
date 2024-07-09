"""Panda-Helper is a simple data-profiling utility for Pandas' DataFrames and Series."""

from __future__ import annotations

from pandahelper.stats import frequency_table, distribution_stats
from pandahelper.profiles import DataFrameProfile, SeriesProfile

__version__ = "0.1.0"
__all__ = ["frequency_table", "distribution_stats", "DataFrameProfile", "SeriesProfile"]
