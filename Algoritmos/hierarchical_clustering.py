import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# Clase HierarchicalClustering
# ============================
class HierarchicalClustering:
    def __init__(self, data=None, method='ward'):
        """
        Inicializa la clase con datos y el método de enlace.
        
        Parámetros:
        - data: np.array con las coordenadas de los puntos
        - method: str, método de enlace ('ward', 'single', 'complete', 'average', etc.)
        """
        if data is None:
            # Datos por defecto si no se pasan en el constructor
            data = np.array([
                [1, 2],
                [1, 4],
                [1, 0],
                [4, 2],
                [4, 4],
                [4, 0]
            ])
        self.data = data
        self.method = method
        self.Z = None
        self.labels = None

    def set_datos(self, data):
        """
        Permite cargar un nuevo conjunto de datos.
        """
        self.data = np.array(data)
        self.Z = None
        self.labels = None

    def calcular_linkage(self):
        """
        Calcula la matriz de enlace Z usando el método especificado.
        """
        self.Z = linkage(self.data, method=self.method)

    def graficar_dendrograma(self, titulo='Hierarchical Clustering Dendrogram', xlabel='Data point', ylabel='Distance'):
        """
        Grafica el dendrograma a partir de la matriz Z.
        """
        if self.Z is None:
            raise ValueError("Debes calcular el linkage antes de graficar. Usa calcular_linkage().")
        
        dendrogram(self.Z)
        plt.title(titulo)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def obtener_clusters(self, n_clusters):
        """
        Corta el dendrograma en n_clusters y devuelve etiquetas para cada punto.
        """
        if self.Z is None:
            raise ValueError("Debes calcular el linkage antes de obtener clusters. Usa calcular_linkage().")
        
        self.labels = fcluster(self.Z, n_clusters, criterion='maxclust')
        return self.labels

    def ejecutar(self, n_clusters=None):
        """
        Ejecuta el flujo completo: calcular linkage, graficar dendrograma
        y (opcionalmente) cortar en n_clusters.
        """
        self.calcular_linkage()
        self.graficar_dendrograma()
        if n_clusters is not None:
            clusters = self.obtener_clusters(n_clusters)
            print(f"Clusters asignados: {clusters}")
            return clusters


# ============================
# Dataset de ejemplo
# ============================
mis_datos = np.array([
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0],
    [10, 10], [10, 11], [11, 10],  # cluster lejano
    [20, 20]  # punto aislado
])

# Guardar dataset como CSV (opcional)
df = pd.DataFrame(mis_datos, columns=["X", "Y"])
df.to_csv("dataset_hierarchical.csv", index=False)
print("Dataset guardado como dataset_hierarchical.csv")


# ============================
# Ejecución del clustering
# ============================
if __name__ == "__main__":
    modelo = HierarchicalClustering(method='ward')
    modelo.set_datos(mis_datos)  # Cargar dataset
    clusters = modelo.ejecutar(n_clusters=3)  # Ejecutar y cortar en 3 clusters
