import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare

class ChiSquareTestAuto:
    def __init__(self, filepath=None, data=None, expected=None, alpha=0.05):
        """
        Inicializa la prueba chi-cuadrado con detección automática del tipo.

        Parámetros:
            filepath (str): ruta al CSV con los datos.
            data (array/list o 2D array/list): datos observados (opcional si no se usa filepath).
            expected (array/list, opcional): frecuencias esperadas (solo para bondad de ajuste).
            alpha (float): nivel de significancia (default 0.05).
        """
        if filepath:
            df = pd.read_csv(filepath, index_col=0)
            df = df.select_dtypes(include=[np.number])  # solo columnas numéricas
            self.data = df.values
            print(f"[ChiSquareTestAuto] Datos cargados desde {filepath} con shape {self.data.shape}")
        else:
            self.data = np.array(data)

        self.alpha = alpha
        self.expected = np.array(expected).flatten() if expected is not None else None
        self.chi2_stat = None
        self.p_value = None
        self.df = None
        self.conclusion = None

        # Detección automática del tipo de prueba
        if self.data.ndim == 1 or self.data.shape[1] == 1:
            self.test_type = 'goodness_of_fit'
        else:
            self.test_type = 'homogeneity'
        print(f"[ChiSquareTestAuto] Tipo de prueba detectado automáticamente: {self.test_type}")

    def run_test(self):
        """Ejecuta la prueba chi-cuadrado según el tipo detectado."""
        if self.test_type == 'goodness_of_fit':
            obs = self.data.flatten()
            
            # Si no hay expected, generar uniforme
            if self.expected is None:
                self.expected = np.full_like(obs, fill_value=np.sum(obs)/len(obs), dtype=float)
            elif len(obs) != len(self.expected):
                raise ValueError(f"Las longitudes de observados ({len(obs)}) y esperados ({len(self.expected)}) no coinciden")
            
            # Ajustar expected para que la suma coincida exactamente
            self.expected = self.expected * (np.sum(obs)/np.sum(self.expected))
            
            self.chi2_stat, self.p_value = chisquare(f_obs=obs, f_exp=self.expected)
            self.df = len(obs) - 1

        elif self.test_type == 'homogeneity':
            chi2, p, dof, _ = chi2_contingency(self.data)
            self.chi2_stat = chi2
            self.p_value = p
            self.df = dof

        else:
            raise ValueError("Tipo de prueba desconocido")

        # decisión
        if self.p_value < self.alpha:
            self.conclusion = "Se rechaza H0: Existe diferencia significativa."
        else:
            self.conclusion = "No se rechaza H0: No hay diferencia significativa."

    def get_results(self):
        if self.chi2_stat is None:
            raise ValueError("Primero ejecuta run_test() antes de obtener resultados.")
        return {
            "Test Type": self.test_type,
            "Chi-Square Statistic": self.chi2_stat,
            "Degrees of Freedom": self.df,
            "P-Value": self.p_value,
            "Alpha": self.alpha,
            "Conclusion": self.conclusion
        }

    def show_results(self):
        results = self.get_results()
        for k, v in results.items():
            print(f"{k}: {v}")


# ============================
# Ejemplo de uso automático
# ============================
if __name__ == "__main__":
    # Homogeneidad detectada automáticamente
    chi_h = ChiSquareTestAuto(filepath="preferencias_bebidas.csv")
    chi_h.run_test()
    chi_h.show_results()

    # Bondad de ajuste detectada automáticamente
    chi_g = ChiSquareTestAuto(filepath="ventas_bondad_ajuste.csv")
    chi_g.run_test()
    chi_g.show_results()
