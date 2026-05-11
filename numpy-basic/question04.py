# Given a pandas DataFrame with columns 'age', 'salary', and 'department'
# write code to:
# (1) extract X as a NumPy array using only 'age' and 'salary'
# (2) extract y as a NumPy array from 'department'

import pandas as pd
import numpy as np
df = pd.DataFrame({
'age': [25, 30, 35, 40, 45],
'salary': [50000, 65000, 80000, 95000, 110000],
'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
})
# Write your code below

X= df[['age','salary']].values
y=df['department'].values

print(X)
print(y)