# 🏦 Credit Risk Scoring: End-to-End MLOps Pipeline

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Este proyecto desarrolla un flujo de trabajo completo (**End-to-End MLOps**) para la evaluación de riesgo crediticio. El sistema predice la probabilidad de impago de un cliente basándose en su perfil financiero y demográfico, utilizando datos del mercado alemán.

El enfoque principal de este repositorio es presentar una **arquitectura de software de Machine Learning robusta, modular y desplegable**, integrando las mejores prácticas de la industria para garantizar la reproducibilidad y escalabilidad. Además, se lleva a cabo un flujo de trabajo de machine learning profesional, desde la limpieza de datos y la selección de modelos mediante **Validación Cruzada Anidada (NCV)** hasta el despliegue en contenedores Docker con interfaces de consumo (API) y explicabilidad (XAI).

---

## 📋 Tabla de Contenidos
- [Arquitectura y Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Automatización (Makefile)](#-automatización-makefile)
- [Instalación y Uso](#-instalación-y-uso)
- [Dashboard & Interpretabilidad](#-dashboard--interpretabilidad-xai)
- [Metodología de ML](#-metodología-de-ml)
- [Resultados del Modelo](#-resultados-del-modelo)
- [Autor](#-autor)

---

## 🛠 Arquitectura y Tech Stack

El proyecto integra herramientas modernas para crear un sistema robusto, modular y escalable:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) (Gestor de paquetes de alto rendimiento).
* **Automatización:** **GNU Make** (Orquestación de comandos).
* **Configuración:** [Hydra](https://hydra.cc/) (Gestión de hiperparámetros vía YAML).
* **Modelado:** XGBoost + Scikit-Learn (Pipelines de preprocesamiento).
* **Optimización:** [Optuna](https://optuna.org/) (Ajuste bayesiano de hiperparámetros).
* **Interpretabilidad (XAI):** [SHAP](https://shap.readthedocs.io/) (Explicación de predicciones "Caja Negra").
* **Interfaces:** * **FastAPI:** API REST para inferencia máquina-a-máquina.
    * **Streamlit:** Dashboard interactivo para usuarios de negocio.
* **Infraestructura:** Docker (Contenedorización completa).

---

## 📂 Estructura del Proyecto

El código sigue una arquitectura de paquete modular, separando configuración, lógica y presentación:

```text
.
├── config/             # ⚙️ Configuración centralizada (Hydra)
├── data/               # 💾 Datos crudos (German Credit Data)
├── docker/             # 🐳 Archivos auxiliares de Docker
├── src/                # 🧠 Código fuente
│   ├── api.py          # API con FastAPI
│   ├── dashboard.py    # Interfaz Web con Streamlit + SHAP
│   ├── pipeline.py     # Definición del modelo y transformadores
│   ├── preprocessing.py# Limpieza e ingeniería de datos
│   └── train.py        # Script de entrenamiento y serialización
├── Dockerfile          # Receta de la imagen de producción
├── Makefile            # 🕹️ Comandos de automatización
├── pyproject.toml      # Dependencias
└── README.md           # Documentación
``` 

---

## 🕹️ Automatización (Makefile)

Para facilitar el uso, el proyecto incluye un Makefile que abstrae los comandos complejos.

```bash
make install	Instala las dependencias con uv.
make train	Ejecuta el pipeline de entrenamiento completo.
make api	Levanta el servidor de la API (FastAPI) en local.
make dashboard	Lanza la aplicación web (Streamlit).
make docker-build	Construye la imagen de Docker.
make docker-run	Ejecuta el contenedor con la App completa.
```

---

## 💻 Instalación y Uso

Tienes dos formas de ejecutar este proyecto: la Profesional (Docker) y la de Desarrollo (Local).


**Opción A: Usando Docker (Recomendado)**

No necesitas instalar Python ni librerías, solo Docker. Garantiza que funcione igual en cualquier máquina. 

1. **Construir y ejecutar:**

```bash
make docker-build
make docker run
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
