from sklearn.preprocessing import RobustScaler
import numpy as np
X_all = np.array([[1.0], [2.0], [3.0], [100.0]]) # 100 is an outlier
scaler = RobustScaler()
scaler.fit(X_all)
print("Median:", scaler.center_)
print("Scale:", scaler.scale_)
X_test = np.array([[2.0]])
print("Transformed:", scaler.transform(X_test).round(3))