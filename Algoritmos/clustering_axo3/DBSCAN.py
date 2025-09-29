
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from axo import Axo, axo_method


class DBSCANClustering(Axo):
    def __init__(self, eps=0.5, min_samples=5, random_state=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.X = None

    def generate_data(self, n_samples=50, centers=3):
        self.X, _ = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=0.8,
            random_state=self.random_state
        )
        print(f"[DBSCANClustering] Generated {n_samples} samples with {centers} centers")

    def set_data(self, X):
        self.X = np.array(X)
        print(f"[DBSCANClustering] Data set with shape {self.X.shape}")

    def apply_dbscan(self, X=None):
        data = X if X is not None else self.X
        if data is None:
            raise ValueError("[apply_dbscan] No data provided")

        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        print(f"[apply_dbscan] Found {n_clusters} clusters")
        return {"labels": labels.tolist(), "n_clusters": n_clusters}
