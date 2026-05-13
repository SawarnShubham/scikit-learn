#This cross-validation code gives inflated scores because of a preprocessing leak.
# Fix it.
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import load_breast_cancer
# X, y = load_breast_cancer(return_X_y=True)
# # Scale ALL data first -- BUG!
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # Cross-validate on already-scaled data
# scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
# print("Scores:", scores.round(4))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

scores = cross_val_score(pipe, X, y, cv=5)

print(scores.round(4))