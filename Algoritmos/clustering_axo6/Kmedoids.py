import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from axo import Axo, axo_method


class KMedoids(Axo):
    def __init__(self, n_clusters=2, max_iter=300, distance_metric='euclidean', random_state=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.medoids = None
        self.labels_ = None
        self.X = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        self.X = X
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
        return self.get_results()


    def predict(self, X, **kwargs):
        X = np.array(X)
        distances = pairwise_distances(X, self.medoids, metric=self.distance_metric)
        return np.argmin(distances, axis=1)

    def get_results(self, **kwargs):
        return {
            "Medoids": self.medoids,
            "Labels": self.labels_,
            "Num Clusters": self.n_clusters,
            "Distance Metric": self.distance_metric
        }


    def plot_clusters(self, **kwargs):
        if self.X is None or self.labels_ is None:
            raise ValueError("Primero ejecuta fit() antes de graficar")
        plt.figure(figsize=(8,6))
        for i in range(self.n_clusters):
            cluster_points = self.X[self.labels_ == i]
            plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f"Cluster {i}")
            plt.scatter(self.medoids[i][0], self.medoids[i][1], marker='x', s=200, c='k')
        plt.title("Clusters K-Medoids")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def run(self, X, **kwargs):
            results = self.fit(X)
            self.plot_clusters()
            return results


