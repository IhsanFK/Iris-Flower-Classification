# main.py

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import os

# Membuat folder plots jika belum ada
os.makedirs("plots", exist_ok=True)

# Memuat dataset Iris
iris = load_iris(as_frame=True)
print("Keys pada dataset Iris:", iris.keys())

# Rename target angka menjadi nama kelas (Setosa, Versicolor, Virginica)
iris.frame["target"] = iris.target_names[iris.target]

# Plot pasangan fitur menggunakan seaborn
sns_plot = sns.pairplot(iris.frame, hue="target")
sns_plot.fig.suptitle("Pairplot Dataset Iris", y=1.02)
sns_plot.savefig("plots/pairplot_iris.png")
plt.clf()

# PCA - Reduksi dimensi dari 4 ke 3 komponen utama
X_reduced = PCA(n_components=3).fit_transform(iris.data)

# Visualisasi hasil PCA dalam ruang 3D
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set(
    title="First Three PCA Dimensions",
    xlabel="1st Eigenvector",
    ylabel="2nd Eigenvector",
    zlabel="3rd Eigenvector",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Tambahkan legenda
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes",
)
ax.add_artist(legend1)

plt.savefig("plots/pca_3d_plot.png")
plt.show()

import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.decomposition import PCA

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)

ax.set(
    title="First three PCA dimensions",
    xlabel="1st Eigenvector",
    ylabel="2nd Eigenvector",
    zlabel="3rd Eigenvector",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Add a legend
legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes",
)
ax.add_artist(legend1)

plt.show()