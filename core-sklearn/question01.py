#Using StandardScaler, write code that fits on X_train, transforms both X_train and X_test, 
# and prints the learned mean and scale. Show that X_train_scaled has mean≈0 and std≈1.

import numpy as np
from sklearn.preprocessing import StandardScaler
X_train = np.array([[10.0, 200.0], [20.0, 400.0], [30.0, 600.0], [40.0, 800.0]])
X_test = np.array([[25.0, 500.0], [15.0, 300.0]])

scaler =StandardScaler()

scaler.fit(X_train)
print("Learned mean:", scaler.mean_)
print("Learned Scale:", scaler.scale_)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) 

print("Train mean after scaling:", X_train_scaled.mean(axis=0).round(4)) # [0. 0.]
print("Train std after scaling: ", X_train_scaled.std(axis=0).round(4)) # [1. 1.]
print("Test scaled:", X_test_scaled.round(3))

final_output = scaler.fit_transform(X_train)
print("Final output after fit_transform on X_test:", final_output.round(3))