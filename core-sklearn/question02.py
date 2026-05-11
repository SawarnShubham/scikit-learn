#This Pipeline crashes when calling predict(). Find and fix the problem.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
X_train = np.random.randn(100, 4)
y_train = (X_train[:, 0] > 0).astype(int)
X_test = np.random.randn(20, 4)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
print(pipe.predict(X_test))

