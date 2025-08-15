import pandas as pd
from sklearn.cluster import KMeans
import os

class KMeansAnalyzer:
    """
    Clase para aplicar el algoritmo KMeans a un conjunto de datos.
    """

    def __init__(self, filepath: str, results_dir: str = "resultados"):
        """
        Constructor de la clase.

        Parámetros:
        - filepath (str): Ruta del archivo CSV que contiene los datos.
        - results_dir (str): Carpeta donde se guardarán los resultados. Por defecto "resultados".
        """
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)  # Crea la carpeta si no existe
        print(f"[KMeansAnalyzer] Archivo cargado con {self.data.shape[0]} filas y {self.data.shape[1]} columnas")

    def aplicar_kmeans(self, columnas: list, n_clusters: int = 3, output_name: str = "kmeans_result.csv"):
        """
        Aplica el algoritmo KMeans y guarda el resultado.

        Parámetros:
        - columnas (list): Lista de nombres de columnas a usar para el clustering.
        - n_clusters (int): Número de clusters a generar. Por defecto 3.
        - output_name (str): Nombre del archivo CSV de salida.
        """
        X = self.data[columnas].dropna()  # Elimina filas con valores nulos
        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        self.data.loc[X.index, "KMeans_Cluster"] = modelo.fit_predict(X)
        ruta_salida = os.path.join(self.results_dir, output_name)
        self.data.to_csv(ruta_salida, index=False)
        print(f"[KMeansAnalyzer] KMeans aplicado. Resultado guardado en: {ruta_salida}")

    def mostrar_columnas(self):
        """Muestra las columnas disponibles en el dataset."""
        print("Columnas disponibles:")
        print(self.data.columns.tolist())

    def ver_datos(self, n: int = 5):
        """Muestra las primeras n filas del dataset."""
        print(self.data.head(n))
