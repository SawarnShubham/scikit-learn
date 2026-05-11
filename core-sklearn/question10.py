#Build a Pipeline for unsupervised learning:
# use StandardScaler followed by KMeans (3 clusters). 
# Fit it on random data, then print the cluster labels for the first 10 samples 
# and the cluster center coordinates.

import numpy as np
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])
pipe.fit(X)
print("Cluster labels for first 10 samples:", pipe.named_steps['kmeans'].labels_[:10])
print("Cluster center coordinates:\n", pipe.named_steps['kmeans'].cluster_centers_.round(3))    