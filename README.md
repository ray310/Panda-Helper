# Quickly inspect Pandas DataFrames and Series with Panda-Helper data profiles
- Perform initial data exploration
- Detect data issues and help with quality control 

### DataFrameProfile:
- Reports DataFrame shape, Series names, and Series data types
- Checks for obvious duplicates
- Provides distribution statistics on null values per row

```
DataFrameProfile(df)
```
![Sample DataFrameProfile](https://github.com/ray310/Panda-Helper/blob/main/images/df_profile.png)


### SeriesProfile:
- Reports data type, number of unique values, and number of null values
- Displays a frequency table of the most and least common values
- Provides distribution statistics (for numeric data)

#### Catgorical data
```
SeriesProfile(df["Direction"])
```
![Sample Categorical SeriesProfile](https://github.com/ray310/Panda-Helper/blob/main/images/series_profile_direction.png)


#### Numeric data
```
SeriesProfile(df["# Vehicles - ETC (E-ZPass)"])
```
![Sample Numeric SeriesProfile](https://github.com/ray310/Panda-Helper/blob/main/images/series_profile_ez.png)


### Using Panda-Helper
- Note that Panda-Helper is not currently a package
- Install any required dependencies to your environment of choice
- Copy `reports.py` (in `src/pandahelper` directory) and incorporate into your analyses
- Cite this repo or let me know if this is helpful


<br><br>Demonstration data obtained from: <br>
https://data.ny.gov/Transportation/Hourly-Traffic-on-Metropolitan-Transportation-Auth/qzve-kjga/data


<br><br>Test data obtained from: <br>
https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
