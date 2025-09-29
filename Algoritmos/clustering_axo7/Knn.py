import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from axo import Axo, axo_method

class KNNAnalyzer(Axo):
    def __init__(self, data=None, filepath: str = None, results_dir: str = "resultados", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data is not None:
            self.data = data.copy()
        elif filepath is not None:
            self.data = pd.read_csv(filepath)
        else:
            self.data = pd.DataFrame()  # Inicializa vacío si no hay dataset
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"[KNNAnalyzer] Dataset cargado con {self.data.shape[0]} filas y {self.data.shape[1]} columnas")


    def aplicar_knn(self, columnas: list, target_col: str, k: int = 5, output_name: str = "knn_result.csv", **kwargs):
        for col in columnas + [target_col]:
            if col not in self.data.columns:
                raise ValueError(f"La columna '{col}' no existe en el dataset.")
        df = self.data.dropna(subset=columnas + [target_col])
        X = df[columnas]
        y = df[target_col]

        if len(df) < k:
            raise ValueError("El número de vecinos (k) es mayor al número de instancias disponibles.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        reporte = classification_report(y_test, y_pred, output_dict=True)

        ruta_salida = os.path.join(self.results_dir, output_name)
        pd.DataFrame(reporte).transpose().to_csv(ruta_salida)
        print(f"[KNNAnalyzer] KNN aplicado. Reporte guardado en: {ruta_salida}")

        return modelo, reporte


    def mostrar_columnas(self, **kwargs):
        print("Columnas disponibles:", self.data.columns.tolist())


    def ver_datos(self, n: int = 5, **kwargs):
        print(self.data.head(n))

    def prueba_rapida(self, n_samples=100, n_features=3, n_classes=3, k=5, **kwargs):
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.rand(n_samples, n_features),
            columns=[f'feature{i+1}' for i in range(n_features)]
        )
        data['target'] = np.random.randint(0, n_classes, size=n_samples)
        self.data = data
        print("[KNNAnalyzer] Dataset de prueba generado.")
        self.ver_datos(5)
        columnas = [f'feature{i+1}' for i in range(n_features)]
        modelo, reporte = self.aplicar_knn(columnas=columnas, target_col='target', k=k)
        print("[KNNAnalyzer] Reporte de clasificación de prueba:")
        print(reporte)
        return modelo, reporte


    def run_test(self, n_samples=100, n_features=3, n_classes=3, k=5, **kwargs):
        
        return self.prueba_rapida(n_samples=n_samples, n_features=n_features, n_classes=n_classes, k=k)

