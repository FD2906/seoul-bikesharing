import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================== 1. DATA VALIDATION ================================================== #

df = pd.read_csv("seoul.csv")

# rows and cols?
print("\n" + "=" * 50 + "\n")
rows, cols = df.shape[0], df.shape[1]
print(f"{rows} rows, {cols} columns")

# info on df
print("\n" + "=" * 50 + "\n")
df.info()

# 1,464 non-null values, 5 float cols, 5 int cols, 4 object cols

# duplicates check 
print("\n" + "=" * 50 + "\n")
print(f"Duplicate rows: {df.duplicated().sum()}")

# nulls check 
print("\n" + "=" * 50 + "\n")
print(f"Rows with null values:\n {df.isnull().sum()}")

# numerical columns
print("\n" + "=" * 50 + "\n")
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(list(numerical_cols)) 
"""
'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 
'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
"""

# I want dates as YYYY-MM
df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)

# ... snowfall might be only values of 0. whats the sum of snowfall column?
print(df['Snowfall (cm)'].sum()) # result of 0
# drop snowfall - not relevant in analysis
df = df.drop('Snowfall (cm)', axis = 1)

# ... 0% humidity doesn't exist. drop all rows with humidity = 0 
df = df[df['Humidity(%)'] != 0]

# summary stats 
print("\n" + "=" * 50 + "\n")
print(df.describe())

# head and tail
print("\n" + "=" * 50 + "\n")
print(df.head())

# ================================================== 2. ANALYSING WEATHER RELATIONSHIPS ================================================== #

# grid of regplots, split into two 2x2 figures. 7 weather variables to plot vs rented bike count. 
weather_cols = [
    'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
    'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)'
]

target = 'Rented Bike Count' # on x-axis for each grid
plot_df = df[[target] + weather_cols].dropna() # drop any null values before plotting

sns.set_style("whitegrid") # white grid seaborn style

def plot_2x2(cols_subset, fig_title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # 2x2 grid with figsize 12,10
    axes = axes.flatten() 
    
    # for each axes/columns in dictionary of axes and subset of columns
    for ax, col in zip(axes, cols_subset):
        # use only non-negative target values for both plotting and fitting
        df_plot = plot_df[[col, target]].dropna()
        df_plot = df_plot[df_plot[target] >= 0]

        sns.regplot(
            data=df_plot,
            x=col,
            y=target,
            ax=ax,
            scatter_kws={'s': 20, 'alpha': 0.6},
            line_kws={'color': 'red'},
            truncate=True  # ensure the fit line is only drawn across the data range
        )
        ax.set_title(f"{target} vs {col}")
        ax.tick_params(axis='x', rotation=25)

        # force y-axis to start at zero
        ax.set_ylim(bottom=0)

    # if fewer than 4 columns, remove extra axes
    for ax in axes[len(cols_subset):]:
        fig.delaxes(ax)
    
    fig.suptitle(fig_title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# first figure: first 4 weather features
plot_2x2(weather_cols[:4], "Rented Bike Count vs Weather (1/2)")

# second figure: next 4 weather features
plot_2x2(weather_cols[4:], "Rented Bike Count vs Weather (2/2)")

# ================================================== 3. TEST STATS ================================================== #

from scipy.stats import pearsonr

print("\n" + "=" * 50 + "\n")

for col in weather_cols:
    # drop nulls for correlation calculation
    df_corr = plot_df[[col, target]].dropna()
    df_corr = df_corr[df_corr[target] >= 0]

    corr_coef, p_value = pearsonr(df_corr[col], df_corr[target])
    print(f"Pearson correlation between {target} and {col}:")
    print(f"  Correlation Coefficient: {corr_coef:.4f}, P-value: {p_value:.4e}\n")


# ================================================== MULTIPLE, ORDINARY LINEAR REGRESSION ================================================== #

# Load and prepare data
df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)  # Convert to 0/1

# SINGLE regression with interaction terms
model = smf.ols('''
    Q('Rented Bike Count') ~ 
    Q('Temperature(°C)') + 
    Q('Rainfall(mm)') + 
    Q('Humidity(%)') +
    is_weekend + 
    Q('Temperature(°C)'):is_weekend + 
    Q('Rainfall(mm)'):is_weekend +
    Q('Humidity(%)'):is_weekend
''', data=df).fit()

print(model.summary())