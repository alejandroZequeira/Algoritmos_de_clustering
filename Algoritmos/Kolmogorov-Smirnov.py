import numpy as np
import pandas as pd
import os
from scipy.stats import kstest

class KolmogorovSmirnovTest:
    def __init__(self, data, dist='norm', alpha=0.05):
        """
        Inicializa la prueba KS.
        
        Parámetros:
            data (array/list): muestra de datos.
            dist (str): distribución de referencia (ejemplo: 'norm').
            alpha (float): nivel de significancia (default 0.05).
        """
        self.data = np.array(data)
        self.dist = dist
        self.alpha = alpha
        self.ks_stat = None
        self.p_value = None
        self.conclusion = None

    def run_test(self):
        """Ejecuta la prueba de Kolmogorov-Smirnov."""
        self.ks_stat, self.p_value = kstest(self.data, self.dist)

        if self.p_value < self.alpha:
            self.conclusion = "Se rechaza H0: La muestra no proviene de la distribución especificada."
        else:
            self.conclusion = "No se rechaza H0: La muestra proviene de la distribución especificada."

    def get_results(self):
        """Devuelve los resultados en un diccionario."""
        if self.ks_stat is None:
            raise ValueError("Primero ejecuta run_test() antes de obtener resultados.")
        
        return {
            "Test": "Kolmogorov-Smirnov Test",
            "Distribution": self.dist,
            "KS Statistic": self.ks_stat,
            "P-Value": self.p_value,
            "Alpha": self.alpha,
            "Conclusion": self.conclusion
        }

    def show_results(self):
        """Imprime los resultados en formato legible."""
        results = self.get_results()
        for k, v in results.items():
            print(f"{k}: {v}")


# ==============================
# PRUEBAS AUTOMÁTICAS
# ==============================
if __name__ == "__main__":
    # Carpeta para guardar resultados
    RESULTS_DIR = "resultados_ks"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Diccionario de datasets simulados
    datasets = {
        "normal_0_1": np.random.normal(loc=0, scale=1, size=100),
        "normal_5_2": np.random.normal(loc=5, scale=2, size=100),
        "uniforme": np.random.uniform(low=-2, high=2, size=100),
        "exponencial": np.random.exponential(scale=1, size=100),
    }

    results = []

    for name, data in datasets.items():
        print(f"\n🔹 Ejecutando prueba KS para dataset: {name}")

        # Normalizamos para comparar con N(0,1) en caso de no serlo
        data_z = (data - np.mean(data)) / np.std(data)

        ks_test = KolmogorovSmirnovTest(data_z, dist="norm", alpha=0.05)
        ks_test.run_test()
        ks_test.show_results()

        results.append(ks_test.get_results() | {"Dataset": name})

    # Exportar resultados
    df_results = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "ks_results.csv")
    df_results.to_csv(output_path, index=False)

    print(f"\n✅ Resultados guardados en: {output_path}")
