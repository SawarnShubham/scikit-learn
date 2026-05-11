from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
# Trying to access coefficients before fitting

model.fit(X, y)
print("Coefficients:", model.coef_)

