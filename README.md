# Algoritmos de Clustering con ASMaC y Axo

Este proyecto implementa algoritmos de clustering utilizando la arquitectura distribuida **AxO**.

---

## Requisitos

- Python >= 3.10
- [Poetry](https://python-poetry.org/)
- Docker (opcional, si se ejecuta el servidor en contenedores)
- Dependencias principales:
  - `axo >=0.0.2`
  - `zmq`
  - `cloudpickle`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

---
como requisito es necesario tener el entorno de axo instalado es decir los contenedores del endpoint listos 
## probar el algoritmo

1. iniciar el entorno en poetry:

```bash
poetry shell
poetry lock
poetry install
```
2. ejecutar prueba:

```bash
python3 tests/test_kmeas.py
```
o 

```bash
poetry run python3 tests/test_kmeas.py
```
## Instalación

1. Clonar el repositorio:

```bash
git clone <tu-repo-url>
cd Algoritmos_de_clustering
```