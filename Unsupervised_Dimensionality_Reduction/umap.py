# UMAP Example
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
reducer = umap.UMAP()
X_umap = reducer.fit_transform(X)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y)
plt.title("UMAP on Digits")
plt.show()
