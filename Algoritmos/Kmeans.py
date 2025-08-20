import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# ========================
# Clase KMeansAnalyzer
# ========================
class KMeansAnalyzer:
    def __init__(self, data=None, filepath: str = None, results_dir: str = "resultados"):
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

# ========================
# Generar dataset de ejemplo
# ========================
np.random.seed(42)
data = pd.DataFrame({
    "X": np.random.rand(30) * 10,
    "Y": np.random.rand(30) * 10
})

# ========================
# Crear instancia y aplicar KMeans
# ========================
kmeans_model = KMeansAnalyzer(data=data)
kmeans_model.mostrar_columnas()
modelo, labels = kmeans_model.aplicar_kmeans(columnas=["X", "Y"], n_clusters=3)
kmeans_model.ver_datos()

# ========================
# Graficar resultados
# ========================
plt.figure(figsize=(8,6))
for cluster in range(3):
    cluster_data = data[labels == cluster]
    plt.scatter(cluster_data["X"], cluster_data["Y"], label=f"Cluster {cluster}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clusters KMeans")
plt.legend()
plt.show()
