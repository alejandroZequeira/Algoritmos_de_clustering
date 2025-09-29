import numpy as np
import pandas as pd
import os
from scipy import stats
from axo import Axo, axo_method

class OneSampleTTest(Axo):
    def __init__(self, data=None, pop_mean=None, alpha=0.05, tail='two-tailed', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.array(data) if data is not None else np.array([])
        self.pop_mean = pop_mean
        self.alpha = alpha
        self.tail = tail.lower()
        self.n = len(self.data)
        self.df = self.n - 1
        self.sample_mean = np.mean(self.data) if self.n > 0 else None
        self.sample_std = np.std(self.data, ddof=1) if self.n > 1 else None
        self.t_stat = None
        self.p_value = None
        self.conclusion = None


    def run_test(self, **kwargs):
        if self.n == 0:
            raise ValueError("No hay datos para ejecutar la prueba.")
        if self.pop_mean is None:
            raise ValueError("Debes proporcionar la media poblacional 'pop_mean'.")
        self.t_stat = (self.sample_mean - self.pop_mean) / (self.sample_std / np.sqrt(self.n))
        if self.tail == 'two-tailed':
            self.p_value = 2 * (1 - stats.t.cdf(abs(self.t_stat), self.df))
        elif self.tail == 'greater':
            self.p_value = 1 - stats.t.cdf(self.t_stat, self.df)
        elif self.tail == 'less':
            self.p_value = stats.t.cdf(self.t_stat, self.df)
        else:
            raise ValueError("El parámetro 'tail' debe ser 'two-tailed', 'greater' o 'less'")

        self.conclusion = "Se rechaza H0: Existe diferencia significativa." if self.p_value < self.alpha else "No se rechaza H0: No hay diferencia significativa."
        return self.get_results()

    def get_results(self, **kwargs):
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

    def show_results(self, **kwargs):
        results = self.get_results()
        for k, v in results.items():
            print(f"{k}: {v}")

    def run_distributed_test(self, **kwargs):
        
            return self.run_test(**kwargs)

if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=5, size=30)
    t_test = OneSampleTTest(data=data, pop_mean=50, alpha=0.05, tail='two-tailed')
    results = t_test.run_distributed_test()
    t_test.show_results()
