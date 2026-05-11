    #This imputation code has a leakage bug. Find and fix it

    # import numpy as np
    # from sklearn.impute import SimpleImputer
    # from sklearn.model_selection import train_test_split
    # X = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0],[4.0, 5.0], [5.0, np.nan], [6.0, 7.0]])
    # y = np.array([0, 1, 0, 1, 0, 1])
    # # Impute ALL data first, then split -- BUG!
    # print("Original Data:" )
    # print(X)
    # imputer = SimpleImputer(strategy='mean')
    # X_imputed = imputer.fit_transform(X)
    # print("\nImputed Data:")
    # print(X_imputed)
    # X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=42)   
    
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = np.array([
    [1.0, np.nan],
    [2.0, 3.0],
    [np.nan, 4.0],
    [4.0, 5.0],
    [5.0, np.nan],
    [6.0, 7.0]
])

y = np.array([0, 1, 0, 1, 0, 1])

print("Original Data:")
print(X)

# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

# Fit only on training data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Transform test using same learned means
X_test_imputed = imputer.transform(X_test)

print("\nImputed Training Data:")
print(X_train_imputed)

print("\nImputed Test Data:")
print(X_test_imputed)