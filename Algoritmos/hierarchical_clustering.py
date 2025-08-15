import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

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

    def ejecutar(self):
        """
        Ejecuta todo el flujo: calcular linkage y graficar dendrograma.
        """
        self.calcular_linkage()
        self.graficar_dendrograma()


# Ejemplo de uso
if __name__ == "__main__":
    # Puedes cambiar método a 'single', 'complete', 'average', etc.
    modelo = HierarchicalClustering(method='ward')
    modelo.ejecutar()
