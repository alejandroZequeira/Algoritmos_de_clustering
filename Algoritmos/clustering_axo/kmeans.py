import pandas as pd

from sklearn.cluster import KMeans
import os
from axo import Axo,axo_method

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
        
    def aplicar_kmeans(self, columnas: list, n_clusters: int = 3, output_name: str = "kmeans_result.csv"):
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

    def mostrar_columnas(self):
        print("Columnas disponibles:", self.data.columns.tolist())
        
    def ver_datos(self, n: int = 5):
        print(self.data.head(n))


