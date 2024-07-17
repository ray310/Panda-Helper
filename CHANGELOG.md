# Changelog

## Unreleased
- Add functionality to perform some common data cleaning tasks.
- Add `geo.py` module and functionality to set 'close' lat-long coordinates to same value.

## 0.1.1 - Unreleased
### Added
- SeriesProfile now reports gaps in pd.Series with type `datetime64` or for Series with `DatetimeIndex`. [gh-20](https://github.com/ray310/Panda-Helper/issues/20)
- `times.py` module has been added with public functions `time_diffs`, `time_diffs_index`, `id_gaps`, `id_gaps_index`. [gh-20](https://github.com/ray310/Panda-Helper/issues/20)
- [`freq_most_least` default parameter for SeriesProfile has been changed to `(10, 5)`.](https://github.com/ray310/Panda-Helper/commit/9ea7a4108996422eaa433e3b86ed20dbbb3c0bdb)

____
## 0.1.0 - 2024-07-14
### Added
- Add memory usage to `DataFrameProfile` [gh-30](https://github.com/ray310/Panda-Helper/issues/30)
- Improve formatting of `distribution_stats` function output [gh-29](https://github.com/ray310/Panda-Helper/issues/29)
- Improved project documentation with project website [gh-2](https://github.com/ray310/Panda-Helper/issues/2)

### Changed
- [Split reports module into `profiles` and `stats`](https://github.com/ray310/Panda-Helper/commit/93320860834e757ab18d86c2b9334efb05738662)
- [Renamed `save_report` method to `save`](https://github.com/ray310/Panda-Helper/commit/876c5f5af8906081f96aff1f1f0ba9d5754a719a)
- [Refactored tests to use pytest fixtures](https://github.com/ray310/Panda-Helper/commit/ff2bf2dd6e73dd4747b62faef4bd350949866a91)

____
## 0.0.4 - 2024-07-09
### Added
- Add support for improved display in Jupyter Notebooks [gh-22](https://github.com/ray310/Panda-Helper/issues/22)
- Add user to select different string formats for profiles [gh-24](https://github.com/ray310/Panda-Helper/issues/24)
- Allow user to specify number of most frequent and least frequent values to display in SeriesProfile [gh-25](https://github.com/ray310/Panda-Helper/issues/25)

____
## 0.0.3 - 2024-07-06
### Added
- Update for Python 3.12
- Switch project build to pyproject.toml [gh-18](https://github.com/ray310/Panda-Helper/issues/18)
- Simplify import: `import pandahelper` now imports `DataFrameProfile`, `SeriesProfile`, `frequency_table`, and `distribution_stats` [gh-17](https://github.com/ray310/Panda-Helper/issues/17)
- Improved `SeriesProfile` to better handle different data types. [gh-19](https://github.com/ray310/Panda-Helper/issues/19)
- Removed excess trailing whitespace on reports [gh-21](https://github.com/ray310/Panda-Helper/issues/21)
____
## 0.0.2 - 2022-06-07
### Added
- Added improved type-checking for functions and profile classes

____
## 0.0.1 - 2022-06-04
### Added
- First version of Panda-Helper
