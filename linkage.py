import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))
print(data)

# ------------------ COMPLETE LINKAGE ------------------

linkage_data = linkage(data, method='complete', metric='euclidean')
dendrogram(linkage_data)

plt.title("Hierarchical Clustering Dendrogram (Complete)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

hc = AgglomerativeClustering(n_clusters=2, linkage='complete')
labels = hc.fit_predict(data)

print("Labels (Complete):", labels)

plt.scatter(x, y, c=labels, cmap='rainbow')
plt.title("Agglomerative Clustering (Complete)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# ------------------ SINGLE LINKAGE ------------------

linkage_data = linkage(data, method='single', metric='euclidean')
dendrogram(linkage_data)

plt.title("Hierarchical Clustering Dendrogram (Single)")
plt.show()

hc = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = hc.fit_predict(data)

print("Labels (Single):", labels)

plt.scatter(x, y, c=labels, cmap='viridis')
plt.title("Agglomerative Clustering (Single)")
plt.show()