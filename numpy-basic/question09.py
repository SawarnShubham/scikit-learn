#The code below tries to select only the rows where age > 30 from a NumPy array. 
# It crashes. Fix it.

import numpy as np
data = np.array([[25, 50000],
[35, 80000],
[28, 45000],
[42, 120000]])
age_mask = data[:, 0] > 30
filtered = data[age_mask == True]
print(filtered)
