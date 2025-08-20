import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

class DBSCANClustering:
    def __init__(self, eps=0.3, min_samples=10, random_state=0):
        """
        Inicializa el modelo DBSCAN.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state

        self.X = None
        self.labels = None
        self.core_samples_mask = None
        self.n_clusters_ = 0

    def generar_datos(self, n_samples=300, centers=4, cluster_std=0.5):
        """
        Genera datos sintéticos de prueba.
        """
        self.X, _ = make_blobs(
            n_samples=n_samples,
            centers=centers,
            cluster_std=cluster_std,
            random_state=self.random_state
        )

    def set_datos(self, X):
        """
        Permite cargar datos propios en lugar de generarlos.
        """
        self.X = np.array(X)

    def entrenar(self):
        """
        Aplica DBSCAN a los datos cargados o generados.
        """
        if self.X is None:
            raise ValueError("No hay datos. Usa generar_datos() o set_datos().")

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
        if self.labels is None:
            raise ValueError("Primero debes entrenar el modelo con entrenar().")

        unique_labels = set(self.labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Ruido → negro

            class_member_mask = (self.labels == k)

            # Puntos núcleo
            xy = self.X[class_member_mask & self.core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',
                     markerfacecolor=tuple(col),
                     markeredgecolor='k',
                     markersize=8)

            # Puntos frontera
            xy = self.X[class_member_mask & ~self.core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',
                     markerfacecolor=tuple(col),
                     markeredgecolor='k',
                     markersize=6)

        plt.title(f"Number of clusters: {self.n_clusters_}")
        plt.show()

    def ejecutar(self, usar_datos_generados=True, **kwargs):
        """
        Ejecuta el flujo completo.
        Si usar_datos_generados=True → genera datos con make_blobs.
        Si usar_datos_generados=False → se espera que ya se hayan cargado datos con set_datos().
        """
        if usar_datos_generados:
            self.generar_datos(**kwargs)
        self.entrenar()
        self.graficar()


# ============================
# Prueba con dataset de ejemplo
# ============================
if __name__ == "__main__":
    # Dataset propio de ejemplo
    mis_datos = np.array([
        [1, 2], [1, 4], [1, 0],
        [4, 2], [4, 4], [4, 0],
        [10, 10], [10, 11], [11, 10],  # cluster lejano
        [20, 20]  # punto aislado (ruido)
    ])

    # Inicializamos el modelo
    modelo = DBSCANClustering(eps=2.0, min_samples=2)

    # Cargamos nuestros datos
    modelo.set_datos(mis_datos)

    # Ejecutamos el modelo (entrenar + graficar)
    modelo.ejecutar(usar_datos_generados=False)
