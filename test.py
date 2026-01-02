import numpy as np

log_value = 1.85
original_mm = np.expm1(log_value)  # expm1 is the inverse of log1p
print(original_mm)

print(np.log1p(5.36))