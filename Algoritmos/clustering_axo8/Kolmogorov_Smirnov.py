import numpy as np
import pandas as pd
import os
from scipy.stats import kstest
from axo import Axo, axo_method


class KolmogorovSmirnovTest(Axo):
    def __init__(self, data=None, dist='norm', alpha=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data is None:
            self.data = np.array([])
        else:
            self.data = np.array(data)
        self.dist = dist
        self.alpha = alpha
        self.ks_stat = None
        self.p_value = None
        self.conclusion = None


    def run_test(self, **kwargs):
        if self.data.size == 0:
            raise ValueError("No hay datos para ejecutar la prueba.")
        self.ks_stat, self.p_value = kstest(self.data, self.dist)
        if self.p_value < self.alpha:
            self.conclusion = "Se rechaza H0: La muestra no proviene de la distribución especificada."
        else:
            self.conclusion = "No se rechaza H0: La muestra proviene de la distribución especificada."
        return self.get_results()


    def get_results(self, **kwargs):
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

 
    def show_results(self, **kwargs):
        results = self.get_results()
        for k, v in results.items():
            print(f"{k}: {v}")


    def run_distributed_test(self, **kwargs):
        
            return self.run_test(**kwargs)


