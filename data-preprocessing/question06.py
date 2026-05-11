#This OneHotEncoder crashes on the test data. Find and fix the issue.
from sklearn.preprocessing import OneHotEncoder
import numpy as np

X_train = np.array([['cat'], ['dog'], ['bird']])
X_test = np.array([['cat'], ['fish']])

ohe = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

ohe.fit(X_train)

print(ohe.transform(X_test))