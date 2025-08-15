import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

class KNNAnalyzer:
    """
    Clase para aplicar el algoritmo KNN a un conjunto de datos.
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
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"[KNNAnalyzer] Archivo cargado con {self.data.shape[0]} filas y {self.data.shape[1]} columnas")

    def aplicar_knn(self, columnas: list, target_col: str, k: int = 5, output_name: str = "knn_result.csv"):
        """
        Aplica el algoritmo KNN y guarda el reporte de clasificación.

        Parámetros:
        - columnas (list): Lista de nombres de columnas a usar como características.
        - target_col (str): Nombre de la columna objetivo (etiqueta).
        - k (int): Número de vecinos a considerar. Por defecto 5.
        - output_name (str): Nombre del archivo CSV de salida.
        """
        # Eliminar filas con valores nulos en las columnas necesarias
        df = self.data.dropna(subset=columnas + [target_col])

        X = df[columnas]  # Características
        y = df[target_col]  # Etiqueta

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crear y entrenar el modelo
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)

        # Predicciones y reporte
        y_pred = modelo.predict(X_test)
        reporte = classification_report(y_test, y_pred, output_dict=True)

        # Guardar reporte
        ruta_salida = os.path.join(self.results_dir, output_name)
        pd.DataFrame(reporte).transpose().to_csv(ruta_salida)
        print(f"[KNNAnalyzer] KNN aplicado. Reporte guardado en: {ruta_salida}")

    def mostrar_columnas(self):
        """Muestra las columnas disponibles en el dataset."""
        print("Columnas disponibles:")
        print(self.data.columns.tolist())

    def ver_datos(self, n: int = 5):
        """Muestra las primeras n filas del dataset."""
        print(self.data.head(n))
