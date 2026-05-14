# Compare Linear Regression, Ridge (alpha=10), and Lasso (alpha=1) 
# on a dataset with 15 features 
# where only 5 are truly relevant.  
# Count how many coefficients each model drives to zero

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
np.random.seed(42)
n, p = 200, 15
X = np.random.randn(n, p)
true_coef = np.array([3, -2, 1.5, -1, 2] + [0]*10)
y = X @ true_coef + np.random.randn(n)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=10)": Ridge(alpha=10),
    "Lasso (alpha=1)": Lasso(alpha=1)
}
for name, model in models.items():
    model.fit(X, y)
    coef = model.coef_
    zero_count = np.sum(coef == 0)
    print(f"{name}: Coefficients: {coef.round(2)}, Zero coefficients: {zero_count}")

