import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from axo import Axo, axo_method



class HierarchicalClustering(Axo):
    def __init__(self, data=None, method='ward', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data is None:
            data = np.array([
                [1, 2], [1, 4], [1, 0],
                [4, 2], [4, 4], [4, 0]
            ])
        self.data = np.array(data)
        self.method = method
        self.Z = None
        self.labels = None

    def set_data(self, data):
        self.data = np.array(data)
        self.Z = None
        self.labels = None

    
    def compute_linkage(self, **kwargs):
        self.Z = linkage(self.data, method=self.method)
        return self.Z

    
    def plot_dendrogram(self, title="Dendrograma de Clustering Jerárquico", xlabel="Puntos", ylabel="Distancia", **kwargs):
        if self.Z is None:
            raise ValueError("Debes calcular linkage primero con compute_linkage().")
        dendrogram(self.Z)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

   
    def get_clusters(self, n_clusters, **kwargs):
        if self.Z is None:
            raise ValueError("Debes calcular linkage primero con compute_linkage().")
        self.labels = fcluster(self.Z, n_clusters, criterion='maxclust')
        return self.labels

   
    def run(self, n_clusters=None, **kwargs):
            self.compute_linkage()
            #self.plot_dendrogram()
            if n_clusters is not None:
                clusters = self.get_clusters(n_clusters)
                print(f"Clusters asignados: {clusters}")
                return clusters

