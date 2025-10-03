import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


iris = load_iris()
X = iris.data

dims = range(1, 8)  # try 1D to 7D
stresses = []

for d in dims:
    mds_nonmetric = MDS(n_components=d, metric=False, random_state=42)
    mds_nonmetric.fit(X)
    stresses.append(mds_nonmetric.stress_)

plt.plot(dims, stresses, marker="o")
plt.xlabel("Number of dimensions")
plt.ylabel("Stress")
plt.title("Non-metric MDS Stress vs Dimensions (Iris dataset)")
plt.show()
