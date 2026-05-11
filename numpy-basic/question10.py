#Write code that creates two 2D NumPy arrays X_train (shape 6x2) and X_test (shape 3x2) manually 
# then verifies they have the correct shapes, correct number of dimensions, and no shared rows.

import numpy as np
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
X_test = np.array([[13, 14], [15, 16], [17, 18]])

# Verify shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Verify number of dimensions
print("X_train dimensions:", X_train.ndim)
print("X_test dimensions:", X_test.ndim)

# Verify no shared rows (simple check for this example)
shared_rows = np.intersect1d(X_train.view([('', X_train.dtype)] * X_train.shape[1]), X_test.view([('', X_test.dtype)] * X_test.shape[1]))
print("Shared rows:", shared_rows)  
