from sklearn.model_selection import train_test_split
import numpy as np
X = np.arange(20).reshape(10, 2)
y = np.arange(10)
X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.3)
X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.3)
X_train3, X_test3, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
X_train4, X_test4, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
print(np.array_equal(X_train1, X_train2))
print(np.array_equal(X_train3, X_train4))