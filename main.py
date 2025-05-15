# main.py

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401
import os

# Membuat folder untuk menyimpan gambar
os.makedirs("plots", exist_ok=True)

# 1. Load dataset Iris
iris = load_iris(as_frame=True)
print("Keys pada dataset Iris:", iris.keys())

# Rename target angka menjadi nama kelas (Setosa, Versicolor, Virginica)
iris.frame["target"] = iris.target_names[iris.target]

# 2. Visualisasi hubungan antar fitur
sns_plot = sns.pairplot(iris.frame, hue="target")
sns_plot.fig.suptitle("Pairplot Dataset Iris", y=1.02)
sns_plot.savefig("plots/pairplot_iris.png")
plt.clf()

# 3. PCA - Reduksi dimensi dari 4 ke 3 komponen utama
X_reduced = PCA(n_components=3).fit_transform(iris.data)

# 4. Visualisasi hasil PCA dalam ruang 3D
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

legend1 = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes",
)
ax.add_artist(legend1)

plt.savefig("plots/pca_3d_plot.png")
plt.show()

# 5. Klasifikasi dengan KNN (tambahan)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 6. Evaluasi model dengan Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Model KNN: {acc:.2f}")