# Demonstrate the difference between fit_transform() and transform().
# Create a StandardScaler, use fit_transform()on training data, 
# then transform() on test data. 
# Print the mean of each column in both transformed datasets.

import numpy as np
from sklearn.preprocessing import StandardScaler
X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
X_test = np.array([[6.0, 60.0], [7.0, 70.0]])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)
print("Mean of each column in X_train_scaled:", X_train_scaled.mean(axis=0).round(4))
print("Mean of each column in X_test_scaled:", X_test_scaled.mean(axis=0).round(4))