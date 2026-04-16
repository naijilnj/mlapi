# Experiment 9
# K-Means Clustering

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data points
x = [5, 8, 12, 5, 4, 13, 15, 7, 10, 14]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Combine into dataset
data = list(zip(x, y))
print(data)

# Apply KMeans
kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit(data)

# Get centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print("Centroids:\n", centroids)
print("Labels:\n", labels)

# Plot data points
plt.scatter(x, y, c=labels, cmap='viridis', label='Data Points')

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    s=100,
    alpha=0.75,
    marker='x',
    label='Centroids'
)

plt.legend()
plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()