# Write code to handle missing values in a mixed dataset: 
# use median imputation for numeric columns and 'Unknown'
# constant for categorical columns.
# Verify no NaN values remain.

import pandas as pd
import numpy as np
df = pd.DataFrame({
'age': [25.0, np.nan, 35.0, 40.0, np.nan],
'income': [50000.0, 60000.0, np.nan, 80000.0, 90000.0],
'city': ['NYC', None, 'LA', 'NYC', None]
})
print("Original DataFrame:")
print(df)
print("\n Missing Values Count:")
print(df.isnull().sum())
# Impute numeric columns with median
numeric_cols = ['age', 'income']
for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Impute categorical columns with 'Unknown'
categorical_cols = ['city']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

print("\nDataFrame after Imputation:")
print(df)
print("\n Missing Values Count after Imputation:")
print(df.isnull().sum())
