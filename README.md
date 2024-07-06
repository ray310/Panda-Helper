![PyPI - Version](https://img.shields.io/pypi/v/panda-helper)
[![Download Stats](https://img.shields.io/pypi/dm/panda-helper)](https://pypistats.org/packages/panda-helper)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/panda-helper)
![Tests Status](https://github.com/ray310/Panda-Helper/actions/workflows/pytest.yml/badge.svg)
![Lint/Format Status](https://github.com/ray310/Panda-Helper/actions/workflows/format_lint.yml/badge.svg)

# Panda-Helper: Quickly and easily inspect data
Panda-Helper is a simple data-profiling utility for Pandas' DataFrames and Series.

Assess data quality and usefulness with minimal effort.

Quickly perform initial data exploration, _so you can move on to more in-depth analysis_.

-----
### DataFrame profiles:
- Report shape
- Detect duplicated rows
- Display series names and data types
- Calculate distribution statistics on null values per row providing a view on data completeness

__Sample DataFrame profile__<br>
_Vehicles passing through toll stations_

    DataFrame-Level Info
    -------------------------  ------------
    DF Shape                   (1586280, 6)
    Duplicated Rows             2184

    Column Name                 Data Type
    --------------------------  -----------
    Plaza ID                    int64
    Date                        object
    Hour                        int64
    Direction                   object
    # Vehicles - ETC (E-ZPass)  int64
    # Vehicles - Cash/VToll     int64

    Summary of Nulls Per Row
    --------------------------  -----------
    count                       1.58628e+06
    min                         0
    1%                          0
    5%                          0
    25%                         0
    50%                         0
    75%                         0
    95%                         0
    99%                         0
    max                         0
    median                      0
    mean                        0
    median absolute deviation   0
    standard deviation          0
    skew                        0

-----
### Series profiles report the:
- Series data type
- Count of non-null values in the series
- Number of unique values
- Count of null values
- Counts and frequency of the most and least common values
- Distribution statistics for numeric-like data

__Sample profile of categorical data__<br>
_Direction vehicles are traveling_

    Direction Info
    ----------------  -------
    Data Type         object
    Count             1586280
    Unique Values     2
    Null Values       0

    Value      Count  % of total
    -------  -------  ------------
    I         814100  51.32%
    O         772180  48.68%

__Sample profile of numeric data__<br>
_Hourly vehicle counts at tolling points_

    # Vehicles - ETC (E-ZPass) Info
    ---------------------------------  -------
    Data Type                          int64
    Count                              1586280
    Unique Values                      8987
    Null Values                        0

      Value    Count  % of total
    -------  -------  ------------
          0     3137  0.20%
         43     1762  0.11%
         44     1743  0.11%
         40     1712  0.11%
         42     1699  0.11%
         41     1682  0.11%
         39     1676  0.11%
         37     1673  0.11%
         48     1659  0.10%
         46     1654  0.10%
         38     1646  0.10%
         45     1641  0.10%
         36     1636  0.10%
         52     1574  0.10%
         47     1572  0.10%
         50     1571  0.10%
         51     1555  0.10%
         53     1547  0.10%
         55     1543  0.10%
         34     1534  0.10%
       8269        1  0.00%
       8438        1  0.00%
       8876        1  0.00%
       8261        1  0.00%
       8694        1  0.00%

    Statistic                            Value
    -------------------------  ---------------
    count                          1.58628e+06
    min                            0
    1%                            25
    5%                            68
    25%                          407
    50%                         1054
    75%                         2071
    95%                         3583
    99%                         6308
    max                        16854
    median                      1054
    mean                        1373.16
    median absolute deviation    751
    standard deviation          1253.1
    skew                           1.69154

-----
### Installing Panda-Helper
`pip install panda-helper`

-----
### Using Panda-Helper
__Profiling a DataFrame__<br>
Create the DataFrameProfile and then display it or save the profile.
```python
import pandas as pd
import pandahelper as ph

data = {
    "user_id": [1, 2, 3, 4, 4],
    "transaction": ["purchase", "return", "purchase", "exchange", "exchange"],
    "amount": [100.00, None, 1400.00, 85.12, 85.12],
    "survey": [None, None, None, "online", "online"],
}
df = pd.DataFrame(data)
df_profile = ph.DataFrameProfile(df)
df_profile
```

    DataFrame-Level Info
    -------------------------  ------
    DF Shape                   (5, 4)
    Obviously Duplicated Rows  1

    Column Name    Data Type
    -------------  -----------
    user_id        int64
    transaction    object
    amount         float64
    survey         object

    Summary of Nulls Per Row
    --------------------------  --------
    count                       5
    min                         0
    1%                          0
    5%                          0
    25%                         0
    50%                         1
    75%                         1
    95%                         1.8
    99%                         1.96
    max                         2
    median                      1
    mean                        0.8
    median absolute deviation   1
    standard deviation          0.83666
    skew                        0.512241

```python
df_profile.save_report("df_profile.txt")
```

__Profiling a Series__<br>
Create the SeriesProfile and then display it or save it. That's it!
```python
series_profile = ph.SeriesProfile(df["amount"])
series_profile
```
    amount Info
    -------------  -------
    Data Type      float64
    Count          4
    Unique Values  3
    Null Values    1

      Value    Count  % of total
    -------  -------  ------------
      85.12        2  50.00%
     100           1  25.00%
    1400           1  25.00%

    Statistic                       Value
    -------------------------  ----------
    count                         4
    min                          85.12
    1%                           85.12
    5%                           85.12
    25%                          85.12
    50%                          92.56
    75%                         425
    95%                        1205
    99%                        1361
    max                        1400
    median                       92.56
    mean                        417.56
    median absolute deviation     7.44
    standard deviation          654.998
    skew                          1.99931

```python
series_profile.save_report("amount_profile.txt")
```
____
### Sample data obtained from:
- https://data.ny.gov/Transportation/Hourly-Traffic-on-Metropolitan-Transportation-Auth/qzve-kjga/data
- https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
