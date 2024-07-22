![PyPI - Version](https://img.shields.io/pypi/v/panda-helper)
[![Download Stats](https://img.shields.io/pypi/dm/panda-helper)](https://pypistats.org/packages/panda-helper)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/panda-helper)
![Tests Status](https://github.com/ray310/Panda-Helper/actions/workflows/pytest.yml/badge.svg)
![Lint/Format Status](https://github.com/ray310/Panda-Helper/actions/workflows/format_lint.yml/badge.svg)

# Panda-Helper: Quickly and easily inspect data
Panda-Helper is a simple, open-source, Python data-profiling utility for Pandas' DataFrames and Series.

Assess data quality and usefulness with minimal effort.

Quickly perform initial data exploration, _so you can move on to more in-depth analysis_.

Please see the [project website](https://ray310.github.io/Panda-Helper/) for more information.

## Installing Panda-Helper
Panda-Helper can be installed with: `pip install panda-helper`.

## Using Panda Helper
For our Panda-Helper tutorial, we are going to use a dataset that counts how many
 bicycles have passed through bike counting sensors at various locations in New York
 City over time. We are going to merge the dataset with some additional metadata for
 the sensors. The datasets can be downloaded from:

- Bicycle Counts: [https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c/about_data](https://data.cityofnewyork.us/Transportation/Bicycle-Counts/uczf-rk3c/about_data)
- Metadata: [https://data.cityofnewyork.us/Transportation/Bicycle-Counters/smn3-rzf9/about_data](https://data.cityofnewyork.us/Transportation/Bicycle-Counters/smn3-rzf9/about_data)

### Loading Data
Download and save data locally.
```Python
import pandas as pd

metadata = pd.read_csv("data/Bicycle_Counters.csv")
bike_counts = pd.read_csv(
    "data/Bicycle_Counts.csv",
    index_col="date",
    parse_dates=["date"],
    date_format="%m/%d/%Y %I:%M:%S %p",
)
bike_counts = bike_counts.join(metadata.set_index("id"), on="id", how="left")
```

### DataFrame Profile
The `DataFrameProfile` is used to get a quick overview of the contents of a Pandas
 DataFrame. It is an object that can be later referenced or saved if desired.
In a single view it provides:

- DataFrame shape.
- Memory usage.
- The number of duplicated rows (if any).
- The datatypes of the individual Series.
- Statistics nulls per row to provide a view on data completeness.
- Time Differences (Diffs or Gaps) if it is a time-indexed DataFrame.
    - In the below example we see that most observations occur at the same time as
   another observation or 15 minutes after the previous observation. There are a few
   gaps where more than 15 minutes has passed since the last observation.


```Python
import pandahelper as ph

ph.DataFrameProfile(bike_counts)
```
```
DataFrame-Level Info
----------------------  -------------
DF Shape                (5589249, 12)
Duplicated Rows         0
Memory Usage (MB)       1,926.950

Series Name    Data Type         Memory Usage (MB)
-------------  --------------  -------------------
Index          datetime64[ns]               44.714
countid        int64                        44.714
id             int64                        44.714
counts         int64                        44.714
status         int64                        44.714
name           object                      438.682
domain         object                      368.89
latitude       float64                      44.714
longitude      float64                      44.714
interval       int64                        44.714
timezone       object                      419.194
sens           int64                        44.714
counter        object                      297.758

Summary of Nulls Per Row
--------------------------  ---------
Number of Columns           12
min                          0
1%                           0
5%                           0
25%                          0
50%                          0
75%                          0
95%                          1
99%                          1
max                          1
mean                         0.240237
standard deviation           0.427228
median                       0
median absolute deviation    0
skew                         1.21604

Time Diffs         Count  % of total
---------------  -------  ------------
0 days 00:00:00  5176050  92.61%
0 days 00:15:00   413183  7.39%
0 days 01:15:00       12  0.00%
0 days 02:15:00        1  0.00%
0 days 00:30:00        1  0.00%
0 days 06:15:00        1  0.00%
```

### Series Profile (Numeric)
The `SeriesProfile` is used to get a quick overview of the contents of a Pandas
 Series. It is an object that can be later referenced or saved if desired.
In a single view it provides:

- Series data type (dtype).
- The number of non-null values.
- The number of unique values.
- The number of null values.
- The counts of some of the most common and least common values in the series which
  can be configured with the optional `freq_most_least` flag
- Distribution statistics for the Series based on the data type.

_Counts are the number of bike crossings at a bike sensor in a window of time_
```Python
ph.SeriesProfile(bike_counts["counts"])
```

```
counts Info
-------------  -------
Data Type      int64
Count          5589249
Unique Values  897
Null Values    0

  Value    Count  % of total
-------  -------  ------------
      0   860809  15.40%
      1   373805  6.69%
      2   279622  5.00%
      3   217329  3.89%
      4   177636  3.18%
      5   150857  2.70%
      6   131232  2.35%
      7   117491  2.10%
      8   106717  1.91%
      9    98373  1.76%
    824        1  0.00%
   1092        1  0.00%
    925        1  0.00%
    894        1  0.00%
   1081        1  0.00%

Statistic                           Value
-------------------------  --------------
count                         5.58925e+06
min                           0
1%                            0
5%                            0
25%                           2
50%                          13
75%                          37
95%                          93
99%                         164
max                        1133
mean                         26.4127
standard deviation           39.3405
median                       13
median absolute deviation    13
skew                          5.17677
```

### Series Profile (Object)
A `SeriesProfile` for an `object` Series will provide similar information as a numeric
 Series but without distribution statistics. Here we use the optional `freq_most_least`
 parameter to show a longer frequency table.

_Name is the designation of the bike sensor station_
```Python
ph.SeriesProfile(bike_counts["name"], freq_most_least=(20, 20))
```
```
name Info
-------------  -------
Data Type      object
Count          5589249
Unique Values  34
Null Values    0

Value                                                          Count  % of total
-----------------------------------------------------------  -------  ------------
Manhattan Bridge Bike Comprehensive                           381148  6.82%
Manhattan Bridge Display Bike Counter                         381148  6.82%
Manhattan Bridge Ped Path                                     368665  6.60%
Ed Koch Queensboro Bridge Shared Path                         368504  6.59%
Williamsburg Bridge Bike Path                                 368433  6.59%
Brooklyn Bridge Bike Path                                     366111  6.55%
Comprehensive Brooklyn Bridge Counter                         365948  6.55%
Staten Island Ferry                                           287203  5.14%
Prospect Park West                                            266080  4.76%
Kent Ave btw North 8th St and North 9th St                    264522  4.73%
Pulaski Bridge                                                243868  4.36%
1st Avenue - 26th St N - Interference testing                 218169  3.90%
Manhattan Bridge 2012 to 2019 Bike Counter                    202785  3.63%
8th Ave at 50th St.                                           195920  3.51%
Manhattan Bridge 2013 to 2018 Bike Counter                    165505  2.96%
Columbus Ave at 86th St.                                      162481  2.91%
Amsterdam Ave at 86th St.                                     162369  2.91%
2nd Avenue - 26th St S                                        136388  2.44%
Brooklyn Bridge Bicycle Path (Roadway)                         95955  1.72%
Kent Ave btw South 6th St. and Broadway                        78478  1.40%
111th St at 50th Ave                                           72567  1.30%
Fountain Ave                                                   63146  1.13%
Willis Ave                                                     62148  1.11%
Willis Ave Bikes                                               62148  1.11%
Willis Ave Peds                                                62148  1.11%
Manhattan Bridge 2012 Test Bike Counter                        36179  0.65%
Manhattan Bridge Interference Calibration 2019 Bike Counter    27675  0.50%
Ocean Pkwy at Avenue J                                         27260  0.49%
Pelham Pkwy                                                    21452  0.38%
Broadway at 50th St                                            20544  0.37%
High Bridge                                                    16276  0.29%
Emmons Ave                                                     16267  0.29%
Forsyth Plaza                                                  14998  0.27%
Concrete Plant Park                                             6761  0.12%
```

### Time Series Functionality
#### Calculate the cumulative gaps in time series data by category
In the above example we saw a notable difference in the number of observations per
 bike counter station. We can use `category_gaps` to check for gaps in
 time-indexed, categorical-like data. We use the `threshold` parameter to define the
 maximum expected increment in the time-indexed data. Some of the bike stations report
 data every 15 minutes and some report data every hour so we can use a threshold of one
 hour.

```Python
ph.category_gaps(bike_counts["name"], threshold=pd.Timedelta(hours=1))
```
```
                                                       Cumulative Gap
Concrete Plant Park                                4234 days 13:45:00
Forsyth Plaza                                      4148 days 16:15:00
Emmons Ave                                         4135 days 12:30:00
High Bridge                                        4135 days 10:15:00
Broadway at 50th St                                4090 days 10:30:00
Pelham Pkwy                                        4081 days 12:15:00
Ocean Pkwy at Avenue J                             4021 days 00:15:00
Manhattan Bridge Interference Calibration 2019 ... 4016 days 15:00:00
Manhattan Bridge 2012 Test Bike Counter            3928 days 01:30:00
Willis Ave Peds                                    3657 days 12:45:00
Willis Ave Bikes                                   3657 days 12:45:00
Willis Ave                                         3657 days 12:45:00
Fountain Ave                                       3647 days 01:45:00
111th St at 50th Ave                               3548 days 21:45:00
Kent Ave btw South 6th St. and Broadway            3487 days 06:30:00
Brooklyn Bridge Bicycle Path (Roadway)             3305 days 06:45:00
2nd Avenue - 26th St S                             2884 days 02:30:00
Amsterdam Ave at 86th St.                          2613 days 09:30:00
Columbus Ave at 86th St.                           2612 days 06:00:00
Manhattan Bridge 2013 to 2018 Bike Counter         2580 days 19:15:00
8th Ave at 50th St.                                2263 days 19:00:00
Manhattan Bridge 2012 to 2019 Bike Counter         2192 days 07:30:00
1st Avenue - 26th St N - Interference testing      2032 days 00:00:00
Pulaski Bridge                                     1764 days 08:45:00
Kent Ave btw North 8th St and North 9th St         1549 days 04:30:00
Prospect Park West                                 1533 days 00:30:00
Staten Island Ferry                                1312 days 22:15:00
Comprehensive Brooklyn Bridge Counter               492 days 13:45:00
Brooklyn Bridge Bike Path                           490 days 21:45:00
Williamsburg Bridge Bike Path                       466 days 15:00:00
Ed Koch Queensboro Bridge Shared Path               465 days 22:45:00
Manhattan Bridge Ped Path                           464 days 07:15:00
Manhattan Bridge Bike Comprehensive                 333 days 14:45:00
Manhattan Bridge Display Bike Counter               333 days 14:45:00
```
#### Identify when gaps occur in time series data
It looks like the 'Manhattan Bridge Bike Comprehensive' category has the smallest
 amount of missing time. We can use `id_gaps_index` to identify when the gaps occur.
 We see that the largest gap for this bike sensor is ~328 days long in 2013.

```Python
mbc = bike_counts["name"][bike_counts["name"] == "Manhattan Bridge Bike Comprehensive"]
ph.id_gaps_index(mbc, threshold=pd.Timedelta(hours=1))
```
```
                                diffs
date  
2013-12-03 00:00:00 328 days 00:15:00
2023-09-27 02:15:00   2 days 02:30:00
2024-01-21 02:15:00   1 days 02:30:00
2023-07-03 02:15:00   1 days 02:30:00
2023-07-01 02:15:00   1 days 02:30:00
2013-12-03 11:00:00   0 days 06:15:00
2012-10-12 15:00:00   0 days 02:15:00
2021-03-14 03:00:00   0 days 01:15:00
2023-03-12 03:00:00   0 days 01:15:00
2022-03-13 03:00:00   0 days 01:15:00
2019-03-10 03:00:00   0 days 01:15:00
2020-03-08 03:00:00   0 days 01:15:00
2018-03-11 03:00:00   0 days 01:15:00
2017-03-12 03:00:00   0 days 01:15:00
2016-03-13 03:00:00   0 days 01:15:00
2015-03-08 03:00:00   0 days 01:15:00
2014-11-04 05:00:00   0 days 01:15:00
2014-03-09 03:00:00   0 days 01:15:00
2024-03-10 03:00:00   0 days 01:15:00
```
