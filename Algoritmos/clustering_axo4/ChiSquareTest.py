import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chisquare
from axo import Axo, axo_method


class ChiSquareTestAuto(Axo):
    def __init__(self, data=None, expected=None, alpha=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = np.array(data)
        self.alpha = alpha
        self.expected = np.array(expected).flatten() if expected is not None else None
        self.chi2_stat = None
        self.p_value = None
        self.df = None
        self.conclusion = None

        if self.data.ndim == 1 or (self.data.ndim == 2 and self.data.shape[1] == 1):
            self.test_type = 'goodness_of_fit'
        else:
            self.test_type = 'homogeneity'

   
    def show_data(self, n=5):
        """Muestra las primeras filas del dataset"""
        print("Datos:")
        print(self.data[:n])
        return self.data[:n].tolist()

 
    def show_info(self):
        """Muestra información básica de la prueba"""
        info = {
            "test_type": self.test_type,
            "alpha": self.alpha,
            "n_obs": self.data.shape
        }
        print("Información de la prueba:", info)
        return info


    def run_test(self):
        """Ejecuta la prueba Chi-Square"""
        if self.test_type == 'goodness_of_fit':
            obs = self.data.flatten()
            if self.expected is None:
                self.expected = np.full_like(obs, fill_value=np.sum(obs)/len(obs), dtype=float)
            elif len(obs) != len(self.expected):
                raise ValueError("Longitudes de observados y esperados no coinciden")
            self.expected = self.expected * (np.sum(obs)/np.sum(self.expected))
            self.chi2_stat, self.p_value = chisquare(f_obs=obs, f_exp=self.expected)
            self.df = len(obs) - 1
        else:  # homogeneity
            chi2, p, dof, _ = chi2_contingency(self.data)
            self.chi2_stat, self.p_value, self.df = chi2, p, dof

        self.conclusion = (
            "Se rechaza H0: Existe diferencia significativa."
            if self.p_value < self.alpha else
            "No se rechaza H0: No hay diferencia significativa."
        )

        result = {
            "chi2": self.chi2_stat,
            "df": self.df,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "conclusion": self.conclusion
        }
        print("Resultado Chi-Square:", result)
        return result

   
