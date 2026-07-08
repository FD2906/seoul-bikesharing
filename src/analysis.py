import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress # linear regression statistical summaries
import statsmodels.formula.api as smf # hypothesis testing 
import statsmodels.api as sm # for multicollinearity testing
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
Goal of the analyse stage: 

1. Use seoul_processed.csv to do further EDA
    - Consider the research question: 
        "Does the effect of weather conditions on bike rental demand differ 
        between Weekdays and Weekends in Seoul's public bike-sharing system?"
    - Create more visualisations: scatterplots, correlation analysis, refining of visualisations

2. Conduct single linear regression between weather variables and bike demand, with supporting regplots 

3. Use scipy.stats or statsmodels to conduct hypothesis testing on our proposed hypothesis: 
    H0: "Weather variables affect bike rental demand equally across Weekdays and Weekends"
    H1: "At least one weather variable exhibits a significantly different effect on bike rental demand between Weekdays and Weekends" 

4. Interpret both testing results and business implications, documenting results in report with selected visualisations.
"""

# ====================================================== 1. Further EDA ====================================================== # 

print("\n" + "=" * 50 + "\n")
df = pd.read_csv("data/seoul_processed.csv")

print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")

print("\n======= Dataset info =======\n")
df.info()

# change date to pd_datetime
df['date'] = pd.to_datetime(df['date'])

print("\n===== Updated dataset info =====\n")
df.info()


# create helper day_type and day name columns
df['day_type'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
df['day_name'] = df['date'].dt.day_name()
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['day_name'] = pd.Categorical(df['day_name'], categories = day_order, ordered = True)

sns.set_theme(style = 'whitegrid') # set whitegrid style 
palette = {'Weekday': '#1f77b4', 'Weekend': '#ff7f0e'} # use clear blue/orange palette pair



# ======= 1.1 linear model plot (lmplot) comparing slopes by day_type - temperature ======= #

temp_lmplot = sns.lmplot(
    data = df,
    x = 'temp',
    y = 'rentals',
    hue = 'day_type',
    palette = palette, # set to palette defined above
    markers = ['o','s'], # o for circle for Weekday, s for square for Weekend
    scatter_kws = {'alpha':0.3, 's':30}, # make more transparent
    height = 8, # set height in inches
    aspect = 1.5, # set width = height * aspect ratio
    legend = False # disable automatic legend so we can plot a custom one
)

ax = temp_lmplot.axes[0,0]

# construct legend handles and labels, place in empty top-left inside axes
handles, labels = ax.get_legend_handles_labels()

if not handles: # if handles are empty, add manually
    import matplotlib.patches as mpatches 
    handles = [mpatches.Patch(color = palette[k]) for k in ['Weekday', 'Weekend']]
    labels = ['Weekday', 'Weekend']

# add legend within axes
ax.legend(
    handles = handles, 
    labels = labels, 
    title = 'Day Type', # title legend box
    loc = 'upper left', # specify location
    bbox_to_anchor = (0.02, 0.98)
)

plt.suptitle('Rentals vs temperature — separate fits - weekday/weekend')
plt.savefig("imgs/eda/1-temp-lmplot-daytype.png", dpi=300)
plt.show()



# ============= 1.2 lmplot comparing slopes by day_type - humidity ============= # 

humidity_lmplot = sns.lmplot(
    data = df,
    x = 'humidity',
    y = 'rentals', 
    hue = 'day_type', 
    palette = palette,
    markers = ['o','s'], 
    scatter_kws = {'alpha':0.3, 's':30},
    height = 8,
    aspect = 1.5,
    legend = False
)

ax = humidity_lmplot.axes[0,0]

# construct legend handles and labels, place in empty top-left inside axes
handles, labels = ax.get_legend_handles_labels()

if not handles: # if handles are empty, add manually
    import matplotlib.patches as mpatches 
    handles = [mpatches.Patch(color = palette[k]) for k in ['Weekday', 'Weekend']]
    labels = ['Weekday', 'Weekend']

# add legend within axes
ax.legend(
    handles = handles, 
    labels = labels, 
    title = 'Day Type', # title legend box
    loc = 'upper left', # specify location
    bbox_to_anchor = (0.02, 0.98)
)

plt.title('Rentals vs humidity — separate fits - Weekday/Weekend')
plt.tight_layout()
plt.savefig("imgs/eda/2-humidity-lmplot-daytype.png", dpi=300)
plt.show()



# ============= 1.3 lmplot comparing slopes by day_type - rainfall_log ============= #

rainfall_lmplot = sns.lmplot(
    data = df,
    x = 'rainfall_log',
    y = 'rentals', 
    hue = 'day_type', 
    palette = palette,
    markers = ['o','s'], 
    scatter_kws = {'alpha':0.3, 's':30},
    height = 8,
    aspect = 1.5,
    legend = False
)

ax = rainfall_lmplot.axes[0,0]

# construct legend handles and labels, place in empty top-left inside axes
handles, labels = ax.get_legend_handles_labels()

if not handles: # if handles are empty, add manually
    import matplotlib.patches as mpatches 
    handles = [mpatches.Patch(color = palette[k]) for k in ['Weekday', 'Weekend']]
    labels = ['Weekday', 'Weekend']

# add legend within axes
ax.legend(
    handles = handles, 
    labels = labels, 
    title = 'Day Type', # title legend box
    loc = 'upper right', # specify location
    bbox_to_anchor = (0.98, 0.98)
)

plt.title('Rentals vs rainfall_log — Separate fits - Weekday/Weekend')
plt.tight_layout()
plt.savefig("imgs/eda/3-rainfall-log-lmplot-daytype.png", dpi=300)
plt.show()



# ============= 1.4 Hourly profile (mean rentals by hour) — Weekday vs Weekend =============

hourly = (
    df.groupby(['hour','day_type'])['rentals'] # grouping on hour and day_type...
    .mean() # averaging on rentals by hour and day_type
    .reset_index() # reset index to turn groupby object back to dataframe
)

plt.figure(figsize = (12, 8))

sns.lineplot(
    data = hourly, 
    x = 'hour', 
    y = 'rentals', 
    hue = 'day_type', # distinguish by weekday/weekend with different colours
    palette = palette, # use defined palette
    marker = 'o'
)

plt.title('Mean Rentals by Hour — Weekday vs Weekend')
plt.xlabel('Hour of day')
plt.xticks(range(0,24)) # set hour ticks from 0 to 23
plt.ylabel('Mean rentals')

plt.tight_layout()
plt.savefig("imgs/eda/4-hourly-profile-daytype.png", dpi=300)
plt.show()



# ============= 1.5 Hour vs. Day-of-week heatmap (mean rentals) =============

pivot = df.pivot_table(
    index = 'hour', 
    columns = 'day_name', 
    values = 'rentals', 
    aggfunc = 'mean' # use the mean rentals for each hour-day_name combination
)
# a pivot table is required to transform the data so that it can be plotted as a heatmap

plt.figure(figsize = (12,8))

sns.heatmap(
    data = pivot, 
    annot = True, # include values
    fmt = '.0f',
    cmap = 'magma', # highest values are brightest, lowest values darkest
    cbar_kws = {'label':'Mean rentals'}, # colorbar label
    linewidths = 0.5, # add lines between each grid
    linecolor = 'gray', # better separates each grid
    annot_kws = {'fontsize': 8}, # smaller font for annotations
    square = False # this will make each cell rectangular rather than square
)

plt.title('Mean Rentals: Hour vs Day of Week')
plt.xlabel('') # no x label - listed day of weeks are intuitive enough
plt.ylabel('Hour')
plt.yticks(rotation = 0) # keep hour labels horizontal for readability


plt.tight_layout()
plt.savefig("imgs/eda/5-hour-dayname-heatmap.png", dpi=300)
plt.show()

# ============= 1.6 Pairplot of selected variables, coloured by day_type =============

subset = df[[ # selected variables
    'rentals', 
    'temp', 
    'humidity', 
    'rainfall_log', 
    'day_type'
]]

sns.pairplot(
    data = subset, 
    hue = 'day_type', 
    palette = palette, 
    kind = 'reg', # linear regressions
    diag_kind = 'kde', # kdes where regressions cannot be plotted
    plot_kws = { 
        'scatter_kws': {'s':15,'alpha':0.3}, # scatterplot made more transparent, markers smaller
        'line_kws': {'lw':1} # standard line size
    }
)

plt.suptitle('Pairwise relationships (selected features) by day type', y = 1.02)
plt.savefig("imgs/eda/6-selected-pairplot.png", dpi = 300)
plt.show()

# ============= 1.7 New selected correlation matrix for collinearity testing =============

print("==================================================")

plt.figure(figsize=(12, 8)) # set figure size

# correlation matrix for weather variables and rented bike count
matrix = df[[
    'temp', 
    'humidity', 
    'rainfall_log'
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
plt.savefig("imgs/eda/7-correlation-heatmap.png")
plt.show()


# ============= 1.8 VIF calculations for multicollinearity testing =============

print("==================================================")

# select predictors to test on
vif_cols = ['temp', 'humidity', 'rainfall_log']

# drop NA rows 
X_vif = df[vif_cols].dropna()

# add constant for VIF calculation 
X_vif_const = sm.add_constant(X_vif)

vif_df = pd.DataFrame({
    'feature': X_vif_const.columns, 
    'VIF': [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
})

# round values for readability, drop constant row
vif_df['VIF'] = vif_df['VIF'].round(3) # rounding
vif_df = vif_df[vif_df['feature'] != 'const'] # drop row

print("\nVariance Inflation Factors (VIF):\n")
print(vif_df.to_string(index=False))

"""
     feature   VIF
        temp 1.284
    humidity 1.526
rainfall_log 1.215

VIF > 4 -> 'further investigation', moderate multicollinearity
VIF > 10 -> 'requires correction', severe multicollinearity

Source: (Pennsylvania State University, 2018)
"""


# ====================================================== 2. Simple linear regression ====================================================== # 

# Find the test statistics between each weather variable and rentals

print("\n" + "=" * 50 + "\n")

weather_cols = ["temp","humidity","rainfall_log"]
target = "rentals"

# implement a streamlined regression summary for the predictors in weather_cols vs. target. 
# use linregress from scipy.stats and find slope, intercept, r2, p-value, std_err for each predictor
# finally plot a complementary regplot for each predictor vs. target

for i, col in enumerate(weather_cols, start=1): # enumerate to get both index (1, 2, 3) and column name
    for day_type in ['Weekday', 'Weekend']: # split the data separately, create separate base dfs to find linregress stats from

        current_df = df[df['day_type'] == day_type] # select weekday / weekend data one at a time

        print(f"\n--- Simple Linear Regression: {col} vs. {target} - {day_type.lower()} ---\n")
        
        # perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(current_df[col], current_df[target])
        
        # round all five statistics to 3 decimal places
        slope = round(slope, 3)
        intercept = round(intercept, 3)
        r2 = round(r_value**2, 3)
        p_value = round(p_value, 3)
        std_err = round(std_err, 3)

        # print summary statistics to terminal
        print(f"Slope: {slope}")
        print(f"Intercept: {intercept}")
        print(f"R-squared: {r_value**2}")
        print(f"P-value: {p_value}")
        print(f"Standard Error: {std_err}")

        # add to a dataframe, then save as csv in a results folder
        results_df = pd.DataFrame({
            'Predictor': [col],
            'Slope': [slope],
            'Intercept': [intercept],
            'R-squared': [r_value**2],
            'P-value': [p_value],
            'Standard Error': [std_err]
        })

        results_df.to_csv(f"results/{i}_linregress_{day_type.lower()}_{col}_vs_{target}.csv", index=False)
        
        # plot regplot
        plt.figure(figsize=(8,6))
        sns.regplot(
            data = current_df,
            x = col,
            y = target,
            scatter_kws = {'alpha':0.3, 's':30},
            line_kws = {'color':'red'}
        )
        plt.title(f'{day_type} rentals vs {col} with linear regression fit') 

        plt.tight_layout()
        plt.savefig(f"imgs/regression/{i}-{day_type.lower()}-{col}-vs-{target}-regplot.png", dpi=300)
        plt.show()

"""
=== Simple Linear Regression: temp vs. rentals - weekday ===

Slope: 72.106
Intercept: -382.495
R-squared: 0.1788473972059598
P-value: 0.0
Standard Error: 4.791

--- Simple Linear Regression: temp vs. rentals - weekend ---

Slope: 96.613
Intercept: -906.402
R-squared: 0.41820396436928886
P-value: 0.0
Standard Error: 5.676

=== Simple Linear Regression: humidity vs. rentals - weekday ===

Slope: -21.775
Intercept: 2517.163
R-squared: 0.2887016703448124
P-value: 0.0
Standard Error: 1.06

--- Simple Linear Regression: humidity vs. rentals - weekend ---

Slope: -20.347
Intercept: 2298.641
R-squared: 0.3461954835250854
P-value: 0.0
Standard Error: 1.393

=== Simple Linear Regression: rainfall_log vs. rentals - weekday ===

Slope: -619.14
Intercept: 1165.914
R-squared: 0.09497980001885938
P-value: 0.0
Standard Error: 59.263

--- Simple Linear Regression: rainfall_log vs. rentals - weekend ---

Slope: -889.502
Intercept: 1179.522
R-squared: 0.13916413669142455
P-value: 0.0
Standard Error: 110.202
"""


# ====================================================== 3. Hypothesis testing ====================================================== #

# using statsmodels to conduct hypothesis testing on our proposed hypothesis:
# H0: "Weather variables affect bike rental demand equally across Weekdays and Weekends"
# H1: "At least one weather variable exhibits a significantly different effect on bike rental demand between Weekdays and Weekends"

print("\n" + "==============================================================================")

# single regression with interaction terms

model = smf.ols('''
    Q('rentals') ~ 
    Q('temp') + 
    Q('rainfall_log') + 
    Q('humidity') +
    is_weekend + 
    Q('temp'):is_weekend + 
    Q('rainfall_log'):is_weekend +
    Q('humidity'):is_weekend
''', data = df).fit()

"""
model explained:

--- mathematical formula: ---
rentals = β₀ +                              ---> intercept
          β₁(temp) + 
          β₂(rainfall_log) + 
          β₃(humidity) + 
          β₄(is_weekend) +                  ---> constant intercept added for weekends
          β₅(temp x is_weekend) + 
          β₆(rainfall_log x is_weekend) + 
          β₇(humidity x is_weekend) +
          ε                                 ---> random error

if is_weekend = 0, rentals = β₀ + 
                      β₁(temp) + 
                      β₂(rainfall_log) + 
                      β₃(humidity)

if is_weekend = 1, rentals = β₀ + 
                      (β₁ + β₅)(temp) + 
                      (β₂ + β₆)(rainfall_log) + 
                      (β₃ + β₇)(humidity) + 
                      β₄(is_weekend)

on weekends, there is an additional temperature effect of β₅(temp), 
rainfall_log effect of β₆(rainfall_log), humidity effect of β₇(humidity)

--- in Python and statmodels OLS: ---
Q('rentals') = target variable
Q('temp') = temperature, a predictor
Q('rainfall_log') = log-transformed rainfall, a predictor
Q('humidity') = humidity, a predictor
is_weekend = weekend dummy variable (0 = weekday, 1 = weekend) - acts as a 'switch' to turn on interaction terms
Q('temp'):is_weekend = interaction term between temperature and weekend status (temperature x weekend)
Q('rainfall_log'):is_weekend = interaction term between rainfall_log and weekend status (rainfall_log x weekend)
Q('humidity'):is_weekend = interaction term between humidity and weekend status (humidity x weekend)
"""

print(model.summary()) # print results

print("\n" + "==============================================================================" + "\n")

print("Standard Error (SE):", round(np.sqrt(model.mse_resid), 3))  # standard error from numpy residual SE, matching excel results

print("\n" + "==============================================================================" + "\n")

"""
Statistical interpretation: 

t-stat: measures how many standard errors the coefficient is away from zero. 
    a higher absolute t-stat indicates a more significant predictor.
    t = coef / std err. 

p-value: tests the null hypothesis that the coefficient is equal to zero (no effect). 
    a low p-value (typically < 0.05) indicates that we can reject the null hypothesis,
    suggesting that the predictor has a statistically significant effect on the target variable.

P > |t|: the p-value for testing H0: this coef = 0. 
    - two-tailed test: |t| means absolute value.
    - model doesn't know your significance level, you decide (typically, a = 0.05, 95% level of significance).
    - most important for analysis.


----------
What p-values mean: 

Main effects (P > |t| = 0.000): highly significant for weekdays. 
Interaction terms: tests whether weekend effect is different from the weekday effect
- Q('temp'):is_weekend -> p = 0.003 -> Weekend slope is significantly different from weekday slope
- Q('humidity'):is_weekend -> p = 0.027 -> Weekend slope is significantly different from weekday slope
- Q('rainfall_log'):is_weekend -> p = 0.709 -> Weekend slope is NOT significantly different from weekday slope 

Corresponds with eda lmplot visualisations above. 
    
----------
Hypothesis focuses on interaction terms - e.g. Q('X'):is_weekend.
Assume a = 0.05, 95% level of significance

Q('temp'):is_weekend -> p = 0.003 -> SIGNIFICANT
Q('humidity'):is_weekend -> p = 0.027 -> SIGNIFICANT
Q('rainfall_log'):is_weekend -> p = 0.709 -> not significant

At least two weather variables show significantly different effects between weekdays/weekends
-> Reject H0.
-> Conclusion: 'at least one weather variable exhibits a significantly different effect on bike rental demand between Weekdays and Weekends'

----------
Other statistical observations...

Why does rainfall_log's weekday/weekend slope (or interaction terms) appear visually different in lmplot, yet statistically not significant?
- Very high standard error in rainfall_log of 123.175 - corresponds with our lmplot, due to large cluster of 0mm values in rainfall_log.
- Small t value of 0.374, far from the critical value of 1.96 at a = 0.05, indicates weak evidence against null hypothesis for rainfall_log interaction term.
- Few rainy days in dataset leads to less reliable estimates for rainfall_log effects.
- Visually, slopes look different due to seaborn lines of best fit. 
- However, these come with large 95% CI bands, indicating uncertainty in slope estimates, matching with std err results. 
"""



"""
Business interpretations and implications: 
(at a = 0.05, 95% level of significance)
(analysis based on n = 1,447 observations from Seoul's bike-sharing system, with n weekdays = 1,042 and n weekends = 405)

Weekdays (simple regression): 
- coef of Q('temp') = 43.91 -> Each time temperature increases by 1°C, 43.91 more bikes are rented.
- coef of Q('humidity') = -14.84 -> Each time humidity goes up by 1%, 14.84 less bikes are rented. 

Weekends (interaction effects): 
- Q('temp'):is_weekend = 25.40, p = 0.003 
    -> Weekend slope = 43.91 + 25.40 = 69.31 rentals/°C
    -> Temperature effect is 25.40 rentals/°C stronger on weekends. 

- Q('humidity'):is_weekend = 4.82, p = 0.027 
    -> Weekend slope = -14.84 + 4.82 = -10.02 rentals/%
    -> Humidity effect is 4.8 rentals/% less negative on weekends (or 4.8 closer to zero)

Log-transformed rainfall is not considered, due to weak evidence and uncertainty of rejecting null hypothesis.


This means...
- On weekends, temperature has an even stronger positive effect on bike rentals compared to weekdays.
- On weekends, humidity has a less negative effect on bike rentals compared to weekdays.
- Rainfall does not significantly differ in effect between weekdays and weekends.

Bikeshare operators can consider...
- Dynamic pricing strategies: increasing per-minute rental rates on warm, less humid weekends to capitalise on higher predicted demand.
- Resource allocation: deploying more bikes to docking stations in anticipation of higher weekend demand during warm, less humid weather.
- Fleet maintenance scheduling: planning maintenance cycles during cooler, more humid periods when demand is lower, especially on weekends.


---

Summary: 

Temperature and humidity both have statistically significant linear relationships with rental counts. 
The positive interaction on weekends indicate temperature's positive effect is stronger on weekends (+25.40 rentals/°C), 
    while humidity's negative effect is reduced on weekends (4.8 rentals/% closer to zero). 
Rainfall's interaction is not significant - so it's effect appears similar across day types. 

Seoul's bikesharing operators can consider the following business strategies: 
1. Increasing per-minute rental rates on warm, less humid weekends to capitalise on higher predicted demand. 
2. Deploying more bikes to docking stations in anticipation of higher weekend demand given warm, less humid weather
3. Planning maintenance cycles during cooler, more humid periods when demand is lower, especially on weekends. 

---


==============================================================================
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           Q('rentals')   R-squared:                       0.393
Model:                            OLS   Adj. R-squared:                  0.390
Method:                 Least Squares   F-statistic:                     132.9
Date:                Tue, 30 Dec 2025   Prob (F-statistic):          6.31e-151
Time:                        14:32:36   Log-Likelihood:                -11315.
No. Observations:                1447   AIC:                         2.265e+04
Df Residuals:                    1439   BIC:                         2.269e+04
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
Intercept                     1191.9283    144.474      8.250      0.000     908.527    1475.330
Q('temp')                       43.9112      4.541      9.670      0.000      35.004      52.819
Q('rainfall_log')             -306.3157     52.457     -5.839      0.000    -409.216    -203.415
Q('humidity')                  -14.8415      1.172    -12.663      0.000     -17.141     -12.542
is_weekend                    -918.0508    265.629     -3.456      0.001   -1439.113    -396.989
Q('temp'):is_weekend            25.3968      8.522      2.980      0.003       8.680      42.113
Q('rainfall_log'):is_weekend    46.0248    123.175      0.374      0.709    -195.596     287.646
Q('humidity'):is_weekend         4.8203      2.179      2.212      0.027       0.546       9.094
==============================================================================
Omnibus:                       86.452   Durbin-Watson:                   0.407
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              101.607
Skew:                           0.649   Prob(JB):                     8.64e-23
Kurtosis:                       2.994   Cond. No.                     1.27e+03
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.27e+03. This might indicate that there are
strong multicollinearity or other numerical problems.
"""