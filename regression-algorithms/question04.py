from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np
np.random.seed(42)
X = np.linspace(0, 5, 50).reshape(-1, 1)
y = 2*X.ravel()**2 - 3*X.ravel() + 1 + np.random.randn(50)*2
for degree in [1, 2, 3, 8]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree)),
    ('lr', LinearRegression())])
pipe.fit(X, y)
train_r2 = pipe.score(X, y)
cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()
print(f"degree={degree}: train R2={train_r2:.3f}, CV R2={cv_r2:.3f}")