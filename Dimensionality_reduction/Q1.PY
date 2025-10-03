import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
import umap.umap_ as umap

digits = load_digits()
X, y = digits.data, digits.target

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=30, method='standard', random_state=42)
X_lle = lle.fit_transform(X)

iso = Isomap(n_components=2, n_neighbors=30)
X_iso = iso.fit_transform(X)

umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = umap_reducer.fit_transform(X)

def plot_embedding(X_embedded, title):
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.tab10, s=10)
    plt.colorbar()
    plt.title(title, fontsize=12)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plot_embedding(X_tsne, "t-SNE (2D)")

plt.subplot(2, 2, 2)
plot_embedding(X_lle, "LLE (2D)")

plt.subplot(2, 2, 3)
plot_embedding(X_iso, "Isomap (2D)")

plt.subplot(2, 2, 4)
plot_embedding(X_umap, "UMAP (2D)")

plt.tight_layout()
plt.show()
