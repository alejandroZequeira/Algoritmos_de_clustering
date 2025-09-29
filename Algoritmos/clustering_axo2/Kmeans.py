import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from axo import Axo, axo_method


class KMeansAnalyzer(Axo):
    def __init__(self, data=None, filepath: str = None, results_dir: str = "resultados", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data is not None:
            self.data = data.copy()
        elif filepath is not None:
            self.data = pd.read_csv(filepath)
        else:
            raise ValueError("Debes proporcionar un DataFrame o un archivo CSV.")
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"[KMeansAnalyzer] Dataset cargado con {self.data.shape[0]} filas y {self.data.shape[1]} columnas")

    
    def apply_kmeans(self, columnas: list, n_clusters: int = 3, output_name: str = "kmeans_result.csv", **kwargs):
        for col in columnas:
            if col not in self.data.columns:
                raise ValueError(f"La columna '{col}' no existe en el dataset.")
        X = self.data[columnas].dropna()
        if len(X) < n_clusters:
            raise ValueError("El número de clusters es mayor que la cantidad de datos disponibles.")

        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        self.data.loc[X.index, "KMeans_Cluster"] = modelo.fit_predict(X)

        ruta_salida = os.path.join(self.results_dir, output_name)
        self.data.to_csv(ruta_salida, index=False)
        print(f"[KMeansAnalyzer] KMeans aplicado. Resultado guardado en: {ruta_salida}")
        return modelo, self.data["KMeans_Cluster"]

   
    def show_columns(self, **kwargs):
        print("Columnas disponibles:", self.data.columns.tolist())

   
    def show_data(self, n: int = 5, **kwargs):
        print(self.data.head(n))

 
    def plot_clusters(self, n_clusters=3, x_col="X", y_col="Y", **kwargs):
        if "KMeans_Cluster" not in self.data.columns:
            raise ValueError("Primero aplica KMeans con apply_kmeans()")
        plt.figure(figsize=(8,6))
        for cluster in range(n_clusters):
            cluster_data = self.data[self.data["KMeans_Cluster"] == cluster]
            plt.scatter(cluster_data[x_col], cluster_data[y_col], label=f"Cluster {cluster}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Clusters KMeans")
        plt.legend()
        plt.show()

 
    """def run(self, columnas: list, n_clusters: int = 3, **kwargs):
        with AxoContextManager.distributed(endpoint_manager=dem()) as dcm:
            self.show_columns()
            modelo, labels = self.apply_kmeans(columnas, n_clusters=n_clusters)
            self.show_data()
            self.plot_clusters(n_clusters=n_clusters)
            return modelo, labels

if __name__ == "__main__":

    np.random.seed(42)
    
    data = pd.DataFrame({
        "X": np.random.rand(30) * 10,
        "Y": np.random.rand(30) * 10
    })

    analyzer = KMeansAnalyzer(data=data)
    
    # Ejecutamos run y obtenemos Result
    res = analyzer.run(columnas=["X", "Y"], n_clusters=3)

    # Comprobamos si Result es OK y extraemos los valores
    if res.is_ok:
        modelo, labels = res.unwrap()
        print("Modelo y labels obtenidos correctamente")
        print("Labels:", labels.tolist() if hasattr(labels, "tolist") else labels)
    else:
        print("Error en run():", res.unwrap_err())"""

