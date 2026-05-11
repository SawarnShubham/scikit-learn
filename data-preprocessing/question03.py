# Create a DataFrame with 3 categorical columns and 2 numeric columns. 
# Apply OneHotEncoder to the categorical columns 
# and StandardScaler to the numeric columns using ColumnTransformer. 
# Print the output shape and feature names

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
df = pd.DataFrame({
'color': ['red', 'blue', 'green', 'red', 'blue'],
'size': ['S', 'M', 'L', 'M', 'S'],
'brand': ['A', 'B', 'A', 'C', 'B'],
'price': [10.0, 20.0, 30.0, 25.0, 15.0],
'weight': [1.5, 2.0, 2.5, 2.2, 1.8]
})

print("Original DataFrame:")
print(df)

# Define the column transformer
df_columns = df.columns
categorical_cols = ['color', 'size', 'brand']
numeric_cols = ['price', 'weight']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)
# Fit and transform the data
X_transformed = preprocessor.fit_transform(df)

print("\nTransformed Data:")
print(X_transformed)

print("\nOutput Shape:")
print(X_transformed.shape)

print("\nFeature Names:")
feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out(numeric_cols).tolist() +
                  preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist())
print(feature_names)