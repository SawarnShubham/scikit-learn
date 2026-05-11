#What does this code output? Why is the test mean NOT zero
from sklearn.preprocessing import StandardScaler
import numpy as np
X_train = np.array([[10.0], [20.0], [30.0]])
X_test = np.array([[40.0], [50.0]])
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Train mean:", X_train_s.mean().round(4))
print("Test mean: ", X_test_s.mean().round(4))
