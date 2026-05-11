#Build a simple Linear Regression model in Python to 
# predict y from X, where X = [1,2,3,4,5] and y = [2,4,6,8,10]. 
# Print the model coefficient and intercept.

import numpy as np

from sklearn.linear_model import LinearRegression
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
print(X)
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(X, y)
print(model.coef_)
print(model.intercept_)