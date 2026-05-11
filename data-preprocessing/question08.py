# Create a complete preprocessing pipeline for a dataset with missing values 
# and categorical features. Use a Pipeline with ColumnTransformer. 
# Fit on train, transform test, and verify shapes.
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 100

df = pd.DataFrame({
    'age': np.random.choice([20, 30, 40, 50, np.nan], n),
    'salary': np.random.choice([50000, 70000, 90000, np.nan], n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', None], n),
    'label': np.random.randint(0, 2, n)
})

numeric_cols = ['age', 'salary']
categorical_cols = ['city']

# Numeric pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit on train
preprocessor.fit(X_train)

# Transform
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("Train shape:", X_train_transformed.shape)
print("Test shape:", X_test_transformed.shape)