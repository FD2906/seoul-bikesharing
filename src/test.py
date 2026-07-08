import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

log_value = 1.85
original_mm = np.expm1(log_value)  # expm1 is the inverse of log1p
print(original_mm)

print(np.log1p(5.36))

df = (pd.read_csv('data/seoul_processed.csv')
      .drop(columns = ['hour', 'rainfall_log', 'is_weekend'], axis = 1)
      .rename(columns = {
          'rainfall': 'rainfall (mm)', 
          'humidity': 'humidity (%)', 
          'temp': 'temp (°C)'
      }))


sns.pairplot(df)

plt.show()