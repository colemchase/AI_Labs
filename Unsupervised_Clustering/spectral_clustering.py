# Spectral Clustering
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

X, _ = make_moons(n_samples=200, noise=0.05)
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors')
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Spectral Clustering")
plt.show()
