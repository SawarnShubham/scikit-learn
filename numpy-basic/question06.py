#Write a Python program to demonstrate the difference 
#between NumPy array addition and Python list concatenation using the + operator.
import numpy as np
a = np.array([1, 2, 3, 4])
b = [1, 2, 3, 4]
print(type(a + a))
print(type(b + b))
print(a + a)
print(b + b)