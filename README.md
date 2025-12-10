# 🏦 Credit Risk Scoring: End-to-End MLOps Pipeline

![Status](https://img.shields.io/badge/status-production-green)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Pipeline MLOps completo para evaluación de riesgo crediticio. El sistema predice la probabilidad de impago basándose en el perfil financiero del cliente, utilizando datos del mercado alemán.

**Arquitectura de ML profesional** con las mejores prácticas: reproducibilidad, escalabilidad, contenedorización Docker, tracking con MLflow, API REST y dashboard interactivo con explicabilidad (SHAP).

---

## 📋 Contenidos
- [Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura](#-estructura-del-proyecto)
- [Automatización](#-automatización-makefile)
- [Instalación](#-instalación-y-uso)
- [Dashboard](#-dashboard--interpretabilidad-xai)
- [Metodología](#-metodología-de-ml)
- [Resultados](#-resultados-del-modelo)
- [Autor](#-autor)

---

## 🛠 Arquitectura y Tech Stack

* **Python 3.12** - Lenguaje principal
* **[uv](https://github.com/astral-sh/uv)** - Gestor de dependencias de alto rendimiento
* **GNU Make** - Automatización de comandos
* **[Hydra](https://hydra.cc/)** - Gestión de configuración vía YAML
* **MLflow** - Tracking de experimentos y versionado de modelos
* **XGBoost + Scikit-Learn** - Modelado y pipelines de preprocesamiento
* **[Optuna](https://optuna.org/)** - Optimización bayesiana de hiperparámetros
* **[SHAP](https://shap.readthedocs.io/)** - Interpretabilidad (XAI)
* **FastAPI** - API REST para inferencia
* **Streamlit** - Dashboard interactivo
* **Docker** - Contenedorización

---

## 📂 Estructura del Proyecto

```text
.
├── config/              # Configuración (Hydra YAML)
├── data/                # Datos crudos (German Credit Data)
├── images/              # Imágenes para documentación
├── mlruns/              # Tracking de experimentos MLflow
├── notebooks/           # Notebooks de análisis e interpretabilidad
├── src/                 # Código fuente
│   ├── api.py           # API REST (FastAPI)
│   ├── dashboard.py     # Dashboard interactivo (Streamlit + SHAP)
│   ├── pipeline.py      # Pipeline de ML (XGBoost + transformadores)
│   ├── preprocessing.py # Limpieza e ingeniería de datos
│   └── train.py         # Entrenamiento y logging a MLflow
├── Dockerfile           # Imagen de producción
├── docker-compose.yml   # Orquestación de servicios
├── Makefile             # Comandos de automatización
└── pyproject.toml       # Dependencias
```

---

## 🕹️ Automatización (Makefile)

| Comando | Descripción |
|---------|-------------|
| `make install` | Instala dependencias con `uv` |
| `make train` | Entrena el modelo y registra en MLflow |
| `make api` | Inicia API REST (FastAPI) en local |
| `make dashboard` | Inicia dashboard (Streamlit) en local |
| `make docker-build` | Construye imagen Docker |
| `make docker-up` | Levanta API + Dashboard en contenedores |
| `make docker-down` | Detiene todos los contenedores |
| `make clean` | Limpia archivos temporales |

---

## 💻 Instalación y Uso

### Opción A: Docker (Recomendado)

```bash
# Construir y arrancar todo el sistema
make docker-up
```

**Acceder a los servicios:**
- 🎨 **Dashboard:** http://localhost:8501
- ⚙️ **API Docs:** http://localhost:8000/docs

```bash
# Detener servicios
make docker-down
```

### Opción B: Ejecución Local

**Requisitos:** Python 3.12+, uv instalado

```bash
# 1. Instalar dependencias
make install

# 2. Entrenar modelo (registra en MLflow)
make train

# 3. Ejecutar servicios
make api        # API en http://localhost:8000
make dashboard  # Dashboard en http://localhost:8501
```

---

## 🧠 Dashboard & Interpretabilidad (XAI)

Dashboard interactivo con Streamlit que proporciona:

1. **Simulación de perfiles** - Formulario intuitivo para datos del cliente
2. **Predicción en tiempo real** - Probabilidad de impago instantánea
3. **Explicabilidad con SHAP** - Visualización de qué variables (edad, historial, saldo) impactan en la decisión del modelo

---

## ⚙️ Metodología de ML

1. **Ingeniería de Datos** - Limpieza, mapeo de variables categóricas (A11 → Saldo Negativo) y normalización de moneda
2. **Pipeline de Preprocesamiento** - ColumnTransformer con escalado numérico y codificación One-Hot
3. **Selección de Modelos** - Validación Cruzada Anidada (Nested CV) para evitar sobreajuste
4. **Optimización** - Búsqueda bayesiana con Optuna maximizando F1-Score
5. **Tracking** - Registro de experimentos, parámetros y modelos en MLflow

---

## 📊 Resultados del Modelo

**XGBoost** seleccionado como modelo de producción por su capacidad para manejar desbalanceo de clases y capturar relaciones no lineales.

![Comparación de modelos mediante Validación Cruzada Anidada](images/ncv_model_comparison.png)

Todos los experimentos están registrados en MLflow con métricas, parámetros y artefactos versionados.

---

## ✒️ Autor

**Juan Pedro García Sanz**

- **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
- **LinkedIn:** [Juan Pedro García Sanz](https://www.linkedin.com/in/juanpedrogarciasanz)

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT.
