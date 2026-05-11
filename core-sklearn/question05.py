#This code has a critical data leakage bug. Identify it and rewrite it correctly
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
X = np.random.randn(200, 5)
y = (X[:, 0] > 0).astype(int)
# Preprocess ALL data first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # BUG IS HERE
# Then split
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))


###updated code
#This code is rewritten to fix the data leakage bug by splitting the data before scaling.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)

X = np.random.randn(200, 5)
y = (X[:, 0] > 0).astype(int)

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Only transform test
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("Test accuracy:", model.score(X_test_scaled, y_test))