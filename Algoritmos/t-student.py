import numpy as np
import pandas as pd
import os
import scipy.stats as stats

class OneSampleTTest:
    def __init__(self, data, pop_mean, alpha=0.05, tail='two-tailed'):
        """
        Inicializa la prueba t de una muestra.
        
        Parámetros:
            data (array/list): datos de la muestra.
            pop_mean (float): media poblacional bajo H0.
            alpha (float): nivel de significancia (default 0.05).
            tail (str): 'two-tailed', 'greater', 'less'.
        """
        self.data = np.array(data)
        self.pop_mean = pop_mean
        self.alpha = alpha
        self.tail = tail.lower()
        self.n = len(self.data)
        self.df = self.n - 1
        self.sample_mean = np.mean(self.data)
        self.sample_std = np.std(self.data, ddof=1)
        self.t_stat = None
        self.p_value = None
        self.conclusion = None

    def run_test(self):
        """Ejecuta la prueba t de Student de una muestra."""
        # calcular estadístico t
        self.t_stat = (self.sample_mean - self.pop_mean) / (self.sample_std / np.sqrt(self.n))

        # calcular valor p según hipótesis
        if self.tail == 'two-tailed':
            self.p_value = 2 * (1 - stats.t.cdf(abs(self.t_stat), self.df))
        elif self.tail == 'greater':  # H1: μ > μ0
            self.p_value = 1 - stats.t.cdf(self.t_stat, self.df)
        elif self.tail == 'less':  # H1: μ < μ0
            self.p_value = stats.t.cdf(self.t_stat, self.df)
        else:
            raise ValueError("El parámetro 'tail' debe ser 'two-tailed', 'greater' o 'less'")

        # decisión
        reject = self.p_value < self.alpha
        self.conclusion = (
            "Se rechaza H0: Existe diferencia significativa."
            if reject else
            "No se rechaza H0: No hay diferencia significativa."
        )

    def get_results(self):
        """Devuelve los resultados en un diccionario."""
        if self.t_stat is None:
            raise ValueError("Primero ejecuta run_test() antes de obtener resultados.")
        
        return {
            "Test": "One-sample t-test",
            "Sample Mean": self.sample_mean,
            "Population Mean (H0)": self.pop_mean,
            "T-Statistic": self.t_stat,
            "Degrees of Freedom": self.df,
            "P-Value": self.p_value,
            "Alpha": self.alpha,
            "Tail": self.tail,
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
    RESULTS_DIR = "resultados_ttest"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Diccionario de pruebas con distintos escenarios
    tests = {
        "media_cercana_a_50": (np.random.normal(loc=50, scale=5, size=30), 50),
        "media_mayor_a_100": (np.random.normal(loc=110, scale=10, size=30), 100),
        "media_menor_a_20": (np.random.normal(loc=15, scale=3, size=30), 20),
    }

    results = []

    for name, (data, mu0) in tests.items():
        print(f"\n🔹 Ejecutando prueba T para dataset: {name}")

        t_test = OneSampleTTest(data, pop_mean=mu0, alpha=0.05, tail='two-tailed')
        t_test.run_test()
        t_test.show_results()

        results.append(t_test.get_results() | {"Dataset": name})

    # Exportar resultados
    df_results = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "ttest_results.csv")
    df_results.to_csv(output_path, index=False)

    print(f"\n✅ Resultados guardados en: {output_path}")
