# Quickly inspect Pandas DataFrames and Series with Panda-Helper reports
- Perform initial data exploration
- Detect data issues and help with quality control 

### DataFrameReports
- Dataframe shape
- Obvious duplicates
- Series names and data types
- Summary statistics on null values per row

```
DataFrameReport(df)
```
![Sample DataFrame Report](https://github.com/ray310/Panda-Helper/blob/main/images/df_report.png)


### SeriesReports
- Data type
- Count of non-nulls in Series
- \# Unique values
- \# Null values
- Frequency table of the most and least common values
- Distribution statistics (for numeric data)

__Catgorical data__
```
SeriesReport(df["Direction"])
```
![Sample Categorical Series Report](https://github.com/ray310/Panda-Helper/blob/main/images/series_report_direction.png)


__Numeric data__
```
SeriesReport(df["# Vehicles - ETC (E-ZPass)"])
```
![Sample Numeric Series Report](https://github.com/ray310/Panda-Helper/blob/main/images/series_report_ez.png)


### Using Panda-Helper
- Note that Panda-Helper is not currently a package
- Install any required dependencies to your environment of choice
- Copy reports.py (in src directory) and incorporate into your analyses
- Cite this repo or let me know if this is helpful


<br><br>Demonstration data obtained from: <br>
https://data.ny.gov/Transportation/Hourly-Traffic-on-Metropolitan-Transportation-Auth/qzve-kjga/data


<br><br>Test data obtained from: <br>
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
