# 🎯 Adult Income Predcition (End-to-End MLOps Pipeline)

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

Este proyecto desarrolla un flujo de trabajo completo (**End-to-End MLOps**) para predecir si una persona gana más de $50,000 anuales basándose en datos demográficos y laborales.

op1:El enfoque principal de este repositorio es demostrar un **flujo de trabajo de machine learning robusto y profesional**, desde la limpieza de datos hasta la selección de modelos mediante **Validación Cruzada Anidada (NCV)** para obtener una estimación de rendimiento imparcial.

op2:El enfoque principal de este repositorio ha evolucionado de un análisis exploratorio a una **arquitectura de software de Machine Learning robusta, modular y desplegable**, integrando las mejores prácticas de la industria para garantizar la reproducibilidad y escalabilidad.

---

## 📋 Tabla de Contenidos
- [Arquitectura y Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación y Uso (Docker & Local)](#-instalación-y-uso)
- [API de Predicción](#-api-de-predicción)
- [Metodología de ML](#-metodología-de-ml)
- [Resultados del Modelo](#-resultados-del-modelo)
- [Autor](#-autor)

---

## 🛠 Arquitectura y Tech Stack

Este proyecto va más allá del modelado tradicional, implementando un ciclo de vida completo:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) (Ultra-rápido y moderno).
* **Configuración:** [Hydra](https://hydra.cc/) (Gestión de hiperparámetros centralizada vía YAML).
* **Modelado:** XGBoost + Scikit-Learn (Pipelines avanzados).
* **Optimización:** [Optuna](https://optuna.org/) (Ajuste bayesiano de hiperparámetros).
* **Tracking:** [MLflow](https://mlflow.org/) (Registro de experimentos y métricas).
* **Despliegue (Serving):** FastAPI + Pydantic (API REST de alto rendimiento).
* **Contenedorización:** Docker (Entorno aislado y reproducible).

---

## 📂 Estructura del Proyecto

Se sigue una estructura de paquete modular:

```text
.
├── config/             # Configuración centralizada (Hydra)
│   └── config.yaml     # Hiperparámetros y rutas
├── data/               # Dataset (adult.csv)
├── docker/             # Archivos auxiliares de Docker
├── notebooks/          # EDA y experimentación inicial (Legacy)
├── src/                # Código fuente modular
│   ├── api.py          # Endpoint de inferencia (FastAPI)
│   ├── pipeline.py     # Construcción del modelo y Sklearn Pipelines
│   ├── preprocessing.py# Limpieza e ingeniería de datos robusta
│   └── train.py        # Script maestro de entrenamiento y serialización
├── Dockerfile          # Definición de la imagen de producción
├── pyproject.toml      # Dependencias del proyecto (uv)
└── README.md           # Documentación
``` 

---

## 🛠 Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Entorno:** Jupyter Notebook / Google Colab

---

## 💻 Instalación y Uso

Tienes dos formas de ejecutar este proyecto: la Profesional (Docker) y la de Desarrollo (Local).

**Opción A: Usando Docker (Recomendado)**

No necesitas instalar Python ni librerías, solo Docker. garantiza que funcione igual en cualquier máquina. 

1. **Construir la imagen:** Descarga dependencias, entrena el modelo y prepara la API automáticamente.


---

## ⚙️ Metodología

El flujo de trabajo sigue los siguientes pasos:

1.  **Análisis Exploratorio (EDA):** Detección de valores nulos (codificados como '?'), análisis de correlaciones y eliminación de duplicados.
2.  **Feature Engineering:** Transformación de la variable objetivo y selección de características numéricas y categóricas.
3.  **Pipeline de Preprocesamiento:**
    * *Numéricas:* Imputación y Estandarización (`StandardScaler`).
    * *Categóricas:* Imputación (moda) y Codificación (`OneHotEncoder`).
4.  **Selección de Modelos (Nested CV):**
    Se probaron múltiples algoritmos. Debido a la complejidad computacional y el tamaño del dataset (~30k muestras), se priorizaron modelos de ensamblaje sobre SVM con kernels no lineales.
5.  **Entrenamiento Final:** El mejor modelo (XGBoost) se re-entrenó con el dataset completo utilizando los hiperparámetros óptimos encontrados.

---

## 📊 Resultados

Tras ejecutar la Validación Cruzada Anidada, se comparó el rendimiento de los modelos utilizando la métrica **F1-Score** (debido al desbalanceo de clases).

| Modelo | F1-Score Medio (NCV) | Desviación Estándar |
| :--- | :--- | :--- |
| **XGBoost** | **0.7220** | +/- 0.008 |
| Random Forest | 0.6785 | +/- 0.012 |
| Regresión Logística | 0.6565 | +/- 0.008 |
| KNN | 0.6290 | +/- 0.011 |

**Visualización de Resultados:**

![Texto alternativo para la imagen](images/ncv_model_comparison.png)

**Conclusión:**
El modelo **XGBoost** demostró ser superior, capturando mejor las relaciones no lineales y manejando eficazmente el desbalanceo de clases gracias al ajuste de `scale_pos_weight`.

---

## 📂 Estructura del Repositorio

```text
.
├── adult.csv                   # Dataset Adult Census Income (Fuente original del proyecto).
├── Proyecto_Adult_Income.ipynb # Notebook principal con el análisis completo (EDA, Preprocesamiento, NCV).
├── pyproject.toml              # Definición de dependencias (para instalación con UV).
├── README.md                   # Documentación del proyecto (este archivo).
├── images/                     # Contiene los gráficos para el README.
└── .gitignore                  # Reglas para ignorar archivos de entorno (.venv, etc.).


```

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juan Pedro García Sanz](https://github.com/Juanpeg1729)
* **LinkedIn:** [linkedin.com/in/juan-pedro-garcía-sanz-443b31343](https://www.linkedin.com/in/juan-pedro-garcía-sanz-443b31343)
