#This code tries to create a 2D feature matrix with 3 samples and 4 features each. 
# It runs but produces the wrong shape. Fix it.
import numpy as np
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
X = np.array(data).reshape(3, 4)
print("Shape:", X.shape) 