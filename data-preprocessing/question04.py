#What does this code print? What is wrong with using LabelEncoder here?
from sklearn.preprocessing import LabelEncoder
import numpy as np
cities = np.array(['Mumbai', 'Delhi', 'Kolkata', 'Mumbai', 'Delhi'])
le = LabelEncoder()
encoded = le.fit_transform(cities)
print(encoded)
print(le.classes_)
print("Delhi - Kolkata =", encoded[1] - encoded[2])