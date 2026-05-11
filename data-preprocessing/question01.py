# Apply StandardScaler and MinMaxScaler to the same dataset.
# Print the min and max of each column after both scalings.
# Explain which scaler keeps all values between 0 and 1.

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X = np.array([[100.0, 0.5], [200.0, 1.5], [300.0, 2.5], [400.0, 3.5], [500.0, 4.5]])

standard_scaler = StandardScaler()
X_standard_scaled = standard_scaler.fit_transform(X)

minmax_scaler = MinMaxScaler()
X_minmax_scaled = minmax_scaler.fit_transform(X)

print("Standard Scaled Data:")
print(X_standard_scaled)
print("\nMinMax Scaled Data:")
print(X_minmax_scaled)

print("\nMin and Max of each column after Standard Scaling:")
print("Column 1: Min =", np.min(X_standard_scaled[:, 0]), "Max =", np.max(X_standard_scaled[:, 0]))
print("Column 2: Min =", np.min(X_standard_scaled[:, 1]), "Max =", np.max(X_standard_scaled[:, 1])) 

print("\nMin and Max of each column after MinMax Scaling:")
print("Column 1: Min =", np.min(X_minmax_scaled[:, 0]), "Max =", np.max(X_minmax_scaled[:, 0]))
print("Column 2: Min =", np.min(X_minmax_scaled[:, 1]), "Max =", np.max(X_minmax_scaled[:, 1])) 
# The MinMaxScaler keeps all values between 0 and 1, as it scales the data to a specified range (default is 0 to 1).