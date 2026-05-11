#You have a dataset with 1000 samples. 
#Write code to split it into train (70%), validation (15%), 
#and test (15%) sets using train_test_split twice.

import numpy as np
from sklearn.model_selection import train_test_split
# Create a dataset with 1000 samples and 10 features
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)  # Binary target variable
# First split: Train (70%) and Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Second split: Validation (15%) and Test (15%) from Temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)       
print("Train set size:", X_train.shape[0])  # Should be 700
print("Validation set size:", X_val.shape[0])  # Should be 150  
print("Test set size:", X_test.shape[0])  # Should be 150
print("Train set class distribution:", np.bincount(y_train))
print("Validation set class distribution:", np.bincount(y_val))
print("Test set class distribution:", np.bincount(y_test))                                  