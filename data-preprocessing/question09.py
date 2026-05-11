#You discover your test set has a feature value of 10,000 
# while all training values were between 0-100. 
# Write code to show how StandardScaler vs RobustScaler
# handle this outlier differently

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
# Sample data with an outlier
X_train = np.array([[10], [20], [30], [40], [50]])
X_test = np.array([[10], [20], [30], [40], [50], [10000]])  # Outlier in test set
# StandardScaler
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test) 
print("StandardScaler - Train:\n", X_train_standard)
print("StandardScaler - Test:\n", X_test_standard)
# RobustScaler
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)
print("RobustScaler - Train:\n", X_train_robust)
print("RobustScaler - Test:\n", X_test_robust)
