#Create two NumPy arrays and demonstrate arithmetic operations, exponentiation, and reshaping 
# while printing their results and shapes.
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print(a * b)
print(a ** 2)
a.reshape(1, -1)
print(a)
print(a.shape, a.reshape(1, -1).shape, a.reshape(-1, 1).shape)