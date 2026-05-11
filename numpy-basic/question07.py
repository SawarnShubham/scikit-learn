#Create a (4, 3) NumPy array of zeros, then replace the second row with the values [7, 8, 9].
# Print the array before and after.

import numpy as np

arr = np.zeros((4, 3), dtype=int)

print("Before:")
print(arr)

arr[1] = [7, 8, 9]

print("After:")
print(arr)