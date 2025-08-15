import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

class DBSCANClustering:
    def __init__(self, eps=0.3, min_samples=10, n_samples=300, centers=4, cluster_std=0.50, random_state=0):
        """
        Inicializa el modelo y los parámetros de generación de datos.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.centers = centers
        self.cluster_std = cluster_std
        self.random_state = random_state

        self.X = None
        self.labels = None
        self.core_samples_mask = None
        self.n_clusters_ = 0

    def generar_datos(self):
        """
        Genera datos de prueba en forma de 'blobs'.
        """
        self.X, _ = make_blobs(
            n_samples=self.n_samples,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=self.random_state
        )

    def entrenar(self):
        """
        Aplica DBSCAN a los datos generados.
        """
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(self.X)

        # Identificar puntos núcleo
        self.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.core_samples_mask[db.core_sample_indices_] = True

        # Guardar etiquetas
        self.labels = db.labels_

        # Calcular número de clusters (ignorando ruido)
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)

    def graficar(self):
        """
        Grafica los clusters detectados por DBSCAN.
        """
        unique_labels = set(self.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))  # Colores automáticos

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Ruido → color negro
                col = [0, 0, 0, 1]

            class_member_mask = (self.labels == k)

            # Puntos núcleo
            xy = self.X[class_member_mask & self.core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',
                     markerfacecolor=tuple(col),
                     markeredgecolor='k',
                     markersize=6)

            # Puntos frontera
            xy = self.X[class_member_mask & ~self.core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',
                     markerfacecolor=tuple(col),
                     markeredgecolor='k',
                     markersize=6)

        plt.title(f"Number of clusters: {self.n_clusters_}")
        plt.show()

    def ejecutar(self):
        """
        Ejecuta el flujo completo: generar datos → entrenar → graficar.
        """
        self.generar_datos()
        self.entrenar()
        self.graficar()


# Ejemplo de uso
if __name__ == "__main__":
    modelo = DBSCANClustering(
        eps=0.3,
        min_samples=10,
        n_samples=300,
        centers=4,
        cluster_std=0.50,
        random_state=0
    )
    modelo.ejecutar()
