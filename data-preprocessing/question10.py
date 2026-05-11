#Apply OrdinalEncoder to a 'rating' column 
# with known order: poor < fair < good < excellent. 
# Then verify the encoding preserves the correct order.


import numpy as np
from sklearn.preprocessing import OrdinalEncoder
ratings = np.array([['good'], ['poor'], ['excellent'], ['fair'], ['good'], ['poor']])


# Define the order of the categories
categories = [['poor', 'fair', 'good', 'excellent']]    
# Create the OrdinalEncoder with the specified categories
ordinal_encoder = OrdinalEncoder(categories=categories)
# Fit and transform the ratings
encoded_ratings = ordinal_encoder.fit_transform(ratings)
print("Encoded Ratings:\n", encoded_ratings)
# Verify the encoding preserves the correct order
print("Category Mapping:")
for category, code in zip(categories[0], range(len(categories[0]))):
    print(f"{category}: {code}")
# The output will show the encoded ratings and the mapping of categories to their corresponding codes, confirming that the order is preserved.

