import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.datasets import load_wine

wine = load_wine()
X, y = wine.data, wine.target

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Metric MDS
mds_metric = MDS(n_components=2, metric=True, random_state=42)
X_mds_metric = mds_metric.fit_transform(X)

# Non-metric MDS
mds_nonmetric = MDS(n_components=2, metric=False, random_state=42)
X_mds_nonmetric = mds_nonmetric.fit_transform(X)

# Plot side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["PCA", "Metric MDS", "Non-metric MDS"]
embeddings = [X_pca, X_mds_metric, X_mds_nonmetric]

for ax, emb, title in zip(axes, embeddings, titles):
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="viridis", alpha=0.7)
    ax.set_title(title)

plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
