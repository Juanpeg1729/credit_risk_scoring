# 🎯 Adult Income Prediction (End-to-End MLOps Pipeline)

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

Este proyecto desarrolla un flujo de trabajo completo (**End-to-End MLOps**) para predecir si una persona gana más de $50,000 anuales basándose en datos demográficos y laborales.

El enfoque principal de este repositorio es presentar una **arquitectura de software de Machine Learning robusta, modular y desplegable**, integrando las mejores prácticas de la industria para garantizar la reproducibilidad y escalabilidad. Además, se lleva a cabo un flujo de trabajo de machine learning profesional, desde la limpieza de datos hasta la selección de modelos mediante **Validación Cruzada Anidada (NCV)** para obtener una estimación de rendimiento imparcial.

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

## 💻 Instalación y Uso

Tienes dos formas de ejecutar este proyecto: la Profesional (Docker) y la de Desarrollo (Local).


**Opción A: Usando Docker (Recomendado)**

No necesitas instalar Python ni librerías, solo Docker. Garantiza que funcione igual en cualquier máquina. 

1. **Construir la imagen:** Descarga dependencias, entrena el modelo y prepara la API automáticamente.

```bash
docker build -t adult-income-app .
```

2. **Ejecutar el contenedor:**

```
docker run -p 8000:8000 adult-income-app
```

3. **Acceder:** abre tu navegador en http://localhost:8000/docs


**Opción B: Ejecución Local (Desarrollo)**

Si deseas editar el código o entrenar manualmente. Requisito: tener uv instalado.

1. **Instalar dependencias:**

```
uv sync
```

2. **Entrenar el modelo (Pipeline completo):** Ejecuta la limpieza, validación y entrenamiento.

```
uv run python -m src.train
```

3. **Levantar la API:**

```
uv run uvicorn src.api:app --reload
```

---

## 🧪 API de Predicción

El proyecto incluye una API REST documentada automáticamente con Swagger UI.

- **Endpoint:** `/predict` (POST)  
- **Input:** JSON con datos demográficos (edad, educación, ocupación, etc.).  
- **Output:** Predicción de clase (`<=50K` o `>50K`) y probabilidad de confianza.

**Ejemplo de uso (Swagger UI):** *(Imagen referencial de la interfaz que verás al lanzar el proyecto)*

---

## 🧠 Metodología de ML

Aunque el código ahora es modular, la lógica de Machine Learning subyacente se mantiene sólida:

1. **Ingeniería de Datos:**
   - Saneamiento de errores de formato (ej. valores corruptos como `5E-1`).
   - Imputación de nulos y eliminación de duplicados.

2. **Pipeline de Preprocesamiento:**
   - `ColumnTransformer` para aplicar escalado (`StandardScaler`) a numéricas y One-Hot Encoding a categóricas.

3. **Selección de Modelos:**
   - Se utilizó **Validación Cruzada Anidada (Nested CV)** para comparar XGBoost, Random Forest y Regresión Logística sin sesgo.

4. **Optimización:**
   - Se implementó **Optuna** para el ajuste fino (fine-tuning) de hiperparámetros del modelo ganador.

---

## 📊 Resultados del Modelo

Tras la evaluación rigurosa, **XGBoost** fue seleccionado como el modelo de producción por su capacidad para manejar desbalanceo y relaciones no lineales.

| Modelo               | F1-Score Medio (NCV) | Desviación |
|---------------------|-----------------------|------------|
| **XGBoost (Optimizado)** | **0.7220**              | +/- 0.008  |
| Random Forest       | 0.6785                | +/- 0.012  |
| Regresión Logística | 0.6565                | +/- 0.008  |

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juan Pedro García Sanz](https://github.com/Juanpeg1729)
* **LinkedIn:** [linkedin.com/in/juan-pedro-garcía-sanz-443b31343](https://www.linkedin.com/in/juan-pedro-garcía-sanz-443b31343)
