import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# =========================
# Clase KMedoids
# =========================
class KMedoids:
    def __init__(self, n_clusters=2, max_iter=300, distance_metric='euclidean', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.medoids = None
        self.labels_ = None

    def fit(self, X):
        X = np.array(X)
        np.random.seed(self.random_state)
        initial_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.medoids = X[initial_idx]

        for _ in range(self.max_iter):
            distances = pairwise_distances(X, self.medoids, metric=self.distance_metric)
            labels = np.argmin(distances, axis=1)
            new_medoids = np.copy(self.medoids)

            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) == 0:
                    continue
                intra_distances = pairwise_distances(cluster_points, cluster_points, metric=self.distance_metric)
                total_distances = np.sum(intra_distances, axis=1)
                new_medoids[i] = cluster_points[np.argmin(total_distances)]

            if np.all(new_medoids == self.medoids):
                break
            self.medoids = new_medoids

        self.labels_ = labels

    def predict(self, X):
        X = np.array(X)
        distances = pairwise_distances(X, self.medoids, metric=self.distance_metric)
        return np.argmin(distances, axis=1)

    def get_results(self):
        return {
            "Medoids": self.medoids,
            "Labels": self.labels_,
            "Num Clusters": self.n_clusters,
            "Distance Metric": self.distance_metric
        }

# =========================
# Generar dataset de prueba
# =========================
np.random.seed(42)
X = np.vstack([
    np.random.normal(loc=[2, 2], scale=0.5, size=(10, 2)),
    np.random.normal(loc=[7, 7], scale=0.5, size=(10, 2)),
    np.random.normal(loc=[2, 7], scale=0.5, size=(10, 2))
])

# =========================
# Crear instancia y entrenar KMedoids
# =========================
kmedoids = KMedoids(n_clusters=3, max_iter=100, random_state=42)
kmedoids.fit(X)

# =========================
# Obtener resultados
# =========================
results = kmedoids.get_results()
print("Medoids:\n", results["Medoids"])
print("Etiquetas asignadas:\n", results["Labels"])

# =========================
# Graficar resultados
# =========================
plt.figure(figsize=(8,6))
for i in range(kmedoids.n_clusters):
    cluster_points = X[results["Labels"] == i]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f"Cluster {i}")
    plt.scatter(results["Medoids"][i][0], results["Medoids"][i][1], marker='x', s=200, c='k')
plt.title("Clusters K-Medoids")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
