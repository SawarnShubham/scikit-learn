#Create a NumPy array from the list [5, 10, 15, 20, 25]. 
# Then print its shape, data type,
# and the result of multiplying every element by 3.


import numpy as np; 

myList = [5, 10, 15, 20, 25];
myarray = np.array(myList);

print(myarray.shape)
print(myarray.dtype)
myarray = myarray*3
print(myarray)