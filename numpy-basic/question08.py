# You receive data as a Python list of lists.
# Each inner list is one sample with 3 features. 
# Convert it to a properly shaped NumPy array for sklearn, verify the shape, 
# and confirm no values are NaN.

import numpy as np

raw_data = [
    [1.2, 3.4, 5.6],
    [7.8, 9.0, float('nan')],
    [2.3, 4.5, 6.7],
    [8.9, 1.1, 3.3]
]

myarr = np.array(raw_data)

print("Original:")
print(myarr)

print("Shape:", myarr.shape)

print("Contains NaN:", np.isnan(myarr).any())

myarr = np.nan_to_num(myarr, nan=0)

print("After replacing NaN:")
print(myarr)

print("Contains NaN:", np.isnan(myarr).any())


