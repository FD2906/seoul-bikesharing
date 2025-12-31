import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('data/seoul.csv')


"""
'prepare' stage activities are as follows: 

1. sanity check of missing values and outliers
2. variable engineering - create weekday/weekend dummy variable using pd.datetime dayofweek
3. enforce data types for relevant variables
    relevant variables are: 
    - 'Date', (pd.datetime)
    - 'Rented Bike Count', (int)
    - 'Hour', (int)
    - 'Temperature(°C)', (float)
    - 'Humidity(%)', (int)
    - 'Rainfall(mm)' (float)
4. record descriptive statistics for relevant numerical variables 
5. initial visualisation - plot the following charts: 
    - correlation charts between weather variables and rented bike count
    - time series line chart of rented bike count over time
        - focus on correct breakdown for day/hours to ensure intuitive patterns are visible
    - bimodal weekday patterns vs. flat weekend patterns (to validate theoretical framing)
6. correlation matrix - understand multicollinearity between temp, humidity, rainfall before running regressions
7. sample split verification - confirm adequate n in both weekday and weekend samples
"""


# ================================================== 1. SANITY CHECKS ================================================== #

# rows and cols?
print("\n" + "=" * 50 + "\n")
rows, cols = df.shape[0], df.shape[1]
print(f"{rows} rows, {cols} columns") # n = 1464. columns = 14

# info on df
print("\n" + "=" * 50 + "\n")
df.info()

# 1,464 non-null values, 5 float cols, 5 int cols, 4 object cols

# convert 'Date' to pd.datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
#print(df['Date'].dtypes) # check conversion

# duplicates check 
print("\n" + "=" * 50 + "\n")
print(f"Duplicate rows: {df.duplicated().sum()}")

# nulls check 
print("\n" + "=" * 50 + "\n")
print(f"Rows with null values:\n {df.isnull().sum()}")

# numerical columns
print("\n" + "=" * 50 + "\n")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print("Numerical columns:")
print(list(numerical_cols)) 
"""
'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 
'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
"""

# head and tail
print("\n" + "=" * 50 + "\n")
print(df.head())

print("\n" + "=" * 50 + "\n")
print(df.tail())

# ================================================== 2. VARIABLE ENGINEERING ================================================== #
# create a 'Weekend' dummy variable - 0 if dayofweek is Mon-Fri, 1 if Sat-Sun
df['Weekend'] = df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

# verify creation
#print("\n" + "=" * 50 + "\n")
#unique_dates = df.drop_duplicates(subset = 'Date', keep = 'first') # print unique Date and Weekend values
#print(unique_dates[['Date', 'Weekend']].head(25)) # weekend functionality works - matches with calendars

# ================================================== 3. ENFORCE VARIABLE TYPES ================================================== #

print("\n" + "=" * 50 + "\n")
print("Current data types:\n")
df.info() # verify current types - match with expectations

# ================================================== 4. RECORD SUMMARY STATISTICS ================================================== #

# summary stats 
print("\n" + "=" * 50 + "\n")
print(df.describe())

"""
Summary stats observations: 

Mean hourly temp = 20.59°C, min = 7.0°C, max = 32.7°C, std = 4.72°C
Mean hourly humidity = 62.45%, min = 0.0%, max = 98.0%, std = 21.13%
Mean hourly rainfall = 0.27mm, min = 0.0mm, max = 3.52mm, std = 1.65mm
"""

# ================================================== 5.1 INITIAL VISUALISATION - CORRELATIONS ================================================== #


# grid of regplots, split into two 2x2 figures. 7 weather variables to plot vs rented bike count. 
weather_cols = [
    'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
    'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)'
]

target = 'Rented Bike Count' # on x-axis for each grid
plot_df = df[[target] + weather_cols].dropna() # drop any null values before plotting

sns.set_style("whitegrid") # set the white grid seaborn style



# create a function to plot 2x2 grid of regplots. following code requires two calls to this function to plot all 7 weather variables.
def plot_2x2(cols_subset, fig_title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 2x2 grid with figsize 12,10
    axes = axes.flatten() 
    
    # for each axes/columns in dictionary of axes and subset of columns
    for ax, col in zip(axes, cols_subset):
        # use only non-negative target values for both plotting and fitting
        df_plot = plot_df[[col, target]].dropna()
        df_plot = df_plot[df_plot[target] >= 0]

        sns.regplot(
            data = df_plot, # selected data to plot (na are all removed)
            x = col,
            y = target,
            ax = ax, # axes
            scatter_kws = {'s': 20, 'alpha': 0.6}, # scatter point size and transparency
            line_kws = {'color': 'red'}, # red regression line
            truncate = True  # ensure the fit line is only drawn across the data range
        )

        ax.set_title(f"{target} vs {col}") # title is 'Rented Bike Count' vs weather column name
        ax.tick_params(axis='x', rotation=25) # rotate x-axis labels for readability
        ax.set_ylim(bottom=0) # force y-axis to start at zero

    # if fewer than 4 columns, remove extra axes
    for ax in axes[len(cols_subset):]:
        fig.delaxes(ax)

    fig.suptitle(fig_title, fontsize = 14, y = 0.98) # overall figure title
    plt.tight_layout(rect = [0, 0, 1, 0.96]) # adjust layout to prevent overlap with suptitle

    return fig # return the figure object - DO NOT SHOW YET - may require further adjustments outside function



# first figure: first 4 weather features 
fig1 = plot_2x2(weather_cols[:4], "Rented Bike Count vs Weather (1/2)")
plt.savefig("imgs/initial-viz/1-demand-vs-weather.png")
plt.show()

# second figure: next 4 weather features
fig2 = plot_2x2(weather_cols[4:], "Rented Bike Count vs Weather (2/2)")
plt.savefig("imgs/initial-viz/2-demand-vs-weather-contd.png")
plt.show()

"""
Note presence of 0% humidity values - likely sensor errors/outliers as unrealistic.
These will be handled in the modelling stage.

Visibility is capped at 2000 (10m) throughout the dataset - likely sensor max.

Solar radiation repeatedly records 0 MJ/m2, potentially even on sunny days - likely sensor errors/outliers as unrealistic.
"""



# ====== 5.2 INITIAL VISUALISATION - BIKE DEMAND TIME SERIES LINE CHART (w/ WEEKDAY/WEEKEND FLAG) ====== #

plt.figure(figsize=(15, 6)) # set new figure size for time series plot

# groupby and aggregate to daily level (sum bikes, keep weekday flag)
daily_bike_counts = (
    df.groupby('Date')
      .agg({'Rented Bike Count': 'sum', 'Weekend': 'first'})  # Weekend: 0 = weekday, 1 = weekend
      .reset_index()
)

# make a friendly label for hue and palette
# palette requires a dictionary mapping of hue values to colors
daily_bike_counts['DayType'] = daily_bike_counts['Weekend'].map({0: 'Weekday', 1: 'Weekend'})
palette = {'Weekday': '#1f77b4', 'Weekend': '#ff7f0e'} # use clear blue/orange palette pair

# draw a light line for trend then overlay colored markers by day type
sns.lineplot(
    data = daily_bike_counts,
    x = 'Date',
    y = 'Rented Bike Count',
    color = 'lightgray', # underlying line will be light gray
    marker = None, # no markers, just line - sns.scatterplot will add markers
    linewidth = 1
)

# the scatterplot overlays colored markers by day type
sns.scatterplot(
    data = daily_bike_counts,
    x = 'Date',
    y = 'Rented Bike Count',
    hue = 'DayType',
    palette = palette, # palette defined above - blue for weekday, orange for weekend
    s = 60, # marker size
    edgecolor = 'black', # marker border colour
    linewidth = 0.3
)

plt.title('Daily Rented Bike Count Over Time')
plt.xlabel('Date')
plt.ylabel('Rented Bike Count')
plt.xticks(rotation=45)

plt.grid(True)
plt.tight_layout()
plt.legend(title='Day type') 

plt.savefig("imgs/initial-viz/3-demand-over-time.png")
plt.show()


# ========================= 5.3 INITIAL VISUALISATION - MEAN HOURLY DEMAND BY WEEKDAY/WEEKEND ========================= #

# here, i want to plot two separate bar charts of mean hourly rented bike counts for weekdays vs weekends
# x-axis should be hour of day (0-23). y-axis should be mean rented bike count
# hue should be weekday vs weekend (0 = weekday, 1 = weekend)

plt.figure(figsize=(14, 6)) # set output figure size

order = list(range(24)) # order of hours from 0 to 23

# find mean series to align y-limits across plots
means_wd = df[df['Weekend'] == 1].groupby('Hour')['Rented Bike Count'].mean()
means_we = df[df['Weekend'] == 0].groupby('Hour')['Rented Bike Count'].mean()

# for both plots, set y-axis limit to 10% above the highest mean value across both plots to ensure consistency
ymax = max(means_wd.max() if not means_wd.empty else 0, means_we.max() if not means_we.empty else 0) * 1.1 


# Weekday plot (left)
ax1 = plt.subplot(1, 2, 1) # ax1 for left plot
sns.barplot(
    data = df[df['Weekend'] == 0], # weekday data
    x = 'Hour',
    y = 'Rented Bike Count',
    order = order, # ascending order of hours 0-23
    errorbar = None, # no error bars
    color = '#1f77b4', # default blue palette for weekdays
    estimator = np.mean, # mean function for bar heights
    ax = ax1 # assign to ax1
)
ax1.set_title('Mean Hourly Rented Bike Count — Weekdays')
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Mean Rented Bike Count')
ax1.set_xticks(range(24))
ax1.set_ylim(0, ymax) # set y-limits consistent with weekend plot

# Weekend plot (right)
ax2 = plt.subplot(1, 2, 2)
sns.barplot(
    data = df[df['Weekend'] == 1], # weekend data
    x = 'Hour',
    y = 'Rented Bike Count',
    order = order,
    errorbar = None, # no error bars
    color = '#ff7f0e', # orange for weekends
    estimator = np.mean,
    ax = ax2
)

ax2.set_title('Mean Hourly Rented Bike Count — Weekends')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('') # no y-label for right plot - shared y-axis label with left plot
ax2.set_xticks(range(24))
ax2.set_ylim(0, ymax) # set y-limits consistent with weekday plot

plt.tight_layout()
plt.savefig("imgs/initial-viz/4-mean-hourly-demand.png")
plt.show()

# ================================================== 6. CORRELATION MATRIX ================================================== #

plt.figure(figsize=(12, 8)) # set figure size

# correlation matrix for weather variables and rented bike count
matrix = df[[
    'Temperature(°C)', 
    'Humidity(%)', 
    'Wind speed (m/s)', 
    'Visibility (10m)', 
    'Dew point temperature(°C)', 
    'Solar Radiation (MJ/m2)', 
    'Rainfall(mm)', 
    'Rented Bike Count'
]].corr()

print("\nCollinearity check among weather variables:\n")

high_collinearity_flag = False

for row in matrix: # for each row in the matrix
    # if a value is > 0.7, print it out as potential multicollinearity issue
    if row != 'Rented Bike Count': # skip target variable
        for col in matrix.columns:
            if col != row and abs(matrix.loc[row, col]) > 0.7: # if col is row then this is 1.0 - skip
                print(f"High correlation between {row} and {col}: {matrix.loc[row, col]:.2f}")
                high_collinearity_flag = True

if not high_collinearity_flag:
    print("No high correlations (> 0.7) found among weather variables.") 

# create a heatmap for this correlation matrix
ax = sns.heatmap(
    matrix, 
    annot = True, 
    cmap = "coolwarm", 
    fmt = ".2f", # present values to 2 decimal places
    linewidths = 0.5
)

# adjust font sizes and rotation
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("imgs/initial-viz/5-correlation-heatmap.png")
plt.show()

# ================================================== 7. SAMPLE SPLIT VERIFICATION ================================================== #
print("\n" + "=" * 50 + "\n")

# grab row counts for each sample
weekday_count = df[df['Weekend'] == 0].shape[0] 
weekend_count = df[df['Weekend'] == 1].shape[0]

print(f"n (weekday): {weekday_count}")
print(f"n (weekend): {weekend_count}")

# n weekday = 1,056
# n weekend = 408
# adequate sample sizes for both groups. 
# watch out for risk of overindexing weekday patterns due to larger sample size.


# one more thing: in process stage and further analysis, select specific columns and RENAME them for clarity. 

"""
Summary of 'prepare' stage activities:

1. Sanity checks completed - no missing values or duplicates found. 
2. 'Weekend' dummy variable created successfully - df['Weekend'] = 0 for Mon-Fri, 1 for Sat-Sun, 
3. Data types enforced as per requirements
4. Descriptive statistics recorded for relevant numerical variables.
    For key weather variables under study: 
        - Mean hourly temp = 20.59°C, min = 7.0°C, max = 32.7°C, std = 4.72°C
        - Mean hourly humidity = 62.45%, min = 0.0%, max = 98.0%, std = 21.13%
        - Mean hourly rainfall = 0.27mm, min = 0.0mm, max = 3.52mm, std = 1.65mm
5. Initial visualisations created:
    - Correlation charts between weather variables and rented bike count plotted.
        - Note presence of 0% humidity values - likely sensor errors/outliers as unrealistic.
            These will be handled in the modelling stage.
        - Visibility is capped at 2000 (10m) throughout the dataset - likely sensor max.
        - Solar radiation repeatedly records 0 MJ/m2, potentially even on sunny days - likely sensor errors/outliers as unrealistic.
    - Time series line chart of rented bike count over time plotted, with weekday/weekend differentiation.
    - Bimodal weekday patterns vs. flat weekend patterns visualised and validated theoretical framing.
        - Found distinct bimodal peaks during weekday rush hours (0700-0900 and 1700-1900), and a smoother distribution on weekends.
6. Correlation matrix plotted - to be analysed for multicollinearity before regression stage.
    Highest bike demand correlations in order: 
        - vs. temperature (0.49)
        - vs. humidity (-0.48)
        - vs. visibility (0.36)
        - vs. wind speed (0.34)
7. Sample split verification completed - both samples adequate for analysis.
    n = 1,056 for weekdays
    n = 408 for weekends
    Need to watch out for risk of overindexing weekday patterns due to larger sample size.
8. Note to self: rename selected columns for clarity in process and analysis stages.


Next up - process stage activities: 

1. Drop unnecessary columns not needed for analysis to streamline dataset.
2. Rename selected columns for clarity in process and analysis stages.
3. Handle outliers/sensor errors in weather variables (e.g., 0% humidity - drop/impute these rows).
4. Consider creating interaction terms if theoretically justified (e.g., temp * humidity).
    - not needed. statsmodels handles this already
5. Save dataset to new CSV for analysis stage.
"""

# ================================================== PROCESS STAGE ACTIVITIES ================================================== #

# ===================== 1. Drop unnecessary columns not needed for analysis to streamline dataset ===================== #
# columns to keep, based on high absolute correlations with rented bike count and theoretical relevance, are:
columns = [
    'Date',
    'Hour',
    'Rented Bike Count',
    'Temperature(°C)',
    'Humidity(%)',
    'Rainfall(mm)',
    'Weekend'
]

df = df[columns] # subset df to only these columns

# ================================ 2. Rename selected columns for clarity in process and analysis stages ================================ #
df = df.rename(columns = {
    'Date': 'date', # lowercase for consistency
    'Hour': 'hour', # hour of day
    'Rented Bike Count': 'rentals', # target variable
    'Temperature(°C)': 'temp', # temperature in celsius
    'Humidity(%)': 'humidity', # humidity in percentage
    'Rainfall(mm)': 'rainfall', # rainfall in mm
    'Weekend': 'is_weekend' # weekend dummy variable
})   

# verify renaming
print("\n" + "=" * 50 + "\n")
print("Renamed columns:\n")
print(df.head())

# ================ 3. Handle outliers/sensor errors in weather variables (e.g., 0% humidity - drop/impute these rows) ================

# ================ 3.1 humidity - 0% humidity doesn't exist. ================
df = df[df['humidity'] != 0] # drop all rows with humidity = 0

# this will reduce n = 1,464 observations, so get a new count of rows across weekdays/weekends
print("\n" + "=" * 50 + "\n")
new_rows = df.shape[0]
print(f"New row count after dropping 0% humidity rows: {new_rows}") # now n = 1,447 rows

# how many weekday vs weekend now?
weekday_count = df[df['is_weekend'] == 0].shape[0] # update weekday_count variable
weekend_count = df[df['is_weekend'] == 1].shape[0] # update weekend_count variable

print(f"n (weekday): {weekday_count}") # now n(weekday) = 1,042
print(f"n (weekend): {weekend_count}") # now n(weekend) = 405

# ================ 3.2. rainfall - consider a log transformation to reduce skewness? ================

"""
current rainfall chart has a large vertical linear cluster (90%) of data at 0mm, and a longer tail of non-zero values.
a log transformation would compress the higher values and spread out the lower values, potentially improving linearity.
log x + 1 is used to avoid errors when log(0) is computed. 

why?
- distribution is heavily right skewed, most values at 0, occasional extreme values (e.g. rainfall = 35mm)
- exponential relationships are made linear
- larger values are compressed, smaller values spread out
- changes in rainfall are interpreted as percentage changes rather than absolute changes

we are testing the 0 -> rain transition in bike rentals, and the magnitude of rain impact on rentals at higher levels
"""

# Log transformation (handles zeros) and round to 3 decimal places
df['rainfall_log'] = np.round(np.log1p(df['rainfall']), 3) # 1 + p is needed to handle 0 values

# use visuals to compare original vs. transformed rainfall distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# before log transformation
axes[0].hist(df['rainfall'], bins=50)
axes[0].set_title('Rainfall Distribution (Original)')
axes[0].set_xlabel('Rainfall (mm)')

# after log transformation
axes[1].hist(df['rainfall_log'], bins=50)
axes[1].set_title('Rainfall Distribution (Log-transformed)')
axes[1].set_xlabel('log (Rainfall + 1)')

# plot adjustments
plt.tight_layout()
plt.savefig('imgs/initial-viz/6-rainfall-transformation.png', dpi=300)
plt.show()

# --------- new regplots to check linearity improvement --------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# before transformation 
sns.regplot(
    data = df, 
    x = 'rainfall', 
    y = 'rentals',
    ax = axes[0], 
    scatter_kws = {'s': 20, 'alpha': 0.3}, 
    line_kws = {'color': 'red'}, 
    truncate = True
)
axes[0].set_title('Original Rainfall')

# after transformation
sns.regplot(
    data = df, 
    x = 'rainfall_log', 
    y = 'rentals', 
    ax = axes[1], 
    scatter_kws = {'s': 20, 'alpha': 0.6}, 
    line_kws = {'color': 'red'}, 
    truncate = True
)
axes[1].set_title('Log-transformed Rainfall')

plt.tight_layout()
plt.savefig('imgs/initial-viz/7-rainfall-linearity-check.png', dpi=300)
plt.show()

# note the linear-log relationship -> Y = untransformed demand, X = log-transformed rainfall.


# ================ 4. Consider creating interaction terms if theoretically justified (e.g., temp * humidity) =============== #

# not needed, as statsmodels handles this already

# ================= 5. Save dataset to new CSV for analysis stage ================= #
df.to_csv('data/seoul_processed.csv', index = False)

print("\n" + "=" * 50 + "\n")
print("Processed dataset saved to 'data/seoul_processed.csv'")
