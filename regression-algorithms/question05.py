# Use RidgeCV and LassoCV to automatically find the best alpha via cross-validation. 
# Compare the best alphas and R² scores

import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
np.random.seed(42)
X = np.random.randn(200, 20)
true_coef = np.array([5,-3,2,0,0,4,-1,0,0,3] + [0]*10)
y = X @ true_coef + np.random.randn(200)*2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
Xtr = sc.fit_transform(X_train)
Xte = sc.transform(X_test)
# RidgeCV: auto-selects best alpha
ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(Xtr, y_train)
print(f"RidgeCV best alpha: {ridge_cv.alpha_}")
print(f"RidgeCV test R2: {ridge_cv.score(Xte, y_test):.4f}")
print(f"Ridge zero coefs: {np.sum(np.abs(ridge_cv.coef_) < 0.01)}")
# LassoCV: auto-selects best alpha
lasso_cv = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(Xtr, y_train)
print(f"LassoCV best alpha: {lasso_cv.alpha_:.5f}")
print(f"LassoCV test R2: {lasso_cv.score(Xte, y_test):.4f}")
print(f"Lasso zero coefs: {np.sum(np.abs(lasso_cv.coef_) < 0.01)}/20")
print(f"Lasso correctly zeroed irrelevant features: {np.sum(lasso_cv.coef_[10:] < 0.01)}/10")