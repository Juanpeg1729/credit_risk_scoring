# 🏦 Credit Risk Scoring con Machine Learning

![Status](https://img.shields.io/badge/status-production-green)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![uv](https://img.shields.io/badge/uv-enabled-purple)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)

Este proyecto implementa un sistema completo para evaluar el riesgo crediticio usando Machine Learning. El modelo analiza el perfil financiero de un cliente y predice automáticamente la probabilidad de impago.

El sistema incluye tracking de experimentos con MLflow, explicabilidad con SHAP, una API REST para predicciones y un dashboard interactivo, todo desplegable con Docker.

---

## 📋 Tabla de Contenidos

- [Arquitectura y Tech Stack](#-arquitectura-y-tech-stack)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Automatización (Makefile)](#%EF%B8%8F-automatización-makefile)
- [Instalación y Uso](#-instalación-y-uso)
- [Dashboard & API](#-dashboard--api)
- [Metodología de Data Science](#-metodología-de-data-science)
- [Resultados](#-resultados)
- [Autor](#-autor)

---

## 🛠 Arquitectura y Tech Stack

El proyecto utiliza tecnologías modernas para crear un sistema robusto y escalable:

* **Lenguaje:** Python 3.11
* **Gestión de Dependencias:** [uv](https://github.com/astral-sh/uv) - Gestor de paquetes de alto rendimiento
* **Configuración:** [Hydra](https://hydra.cc/) - Gestión de configuración mediante YAML
* **Modelo de ML:** 
    * **XGBoost:** Modelo de Gradient Boosting
    * **Scikit-Learn:** Pipelines de preprocesamiento
    * **Optuna:** Optimización de hiperparámetros
* **Tracking:** MLflow - Registro de experimentos y modelos
* **Interpretabilidad:** [SHAP](https://shap.readthedocs.io/) - Explicaciones visuales de predicciones
* **Interfaces:** 
    * **FastAPI:** API REST para predicciones
    * **Streamlit:** Dashboard interactivo con explicabilidad
* **Despliegue:** Docker y Docker Compose

---

## 📂 Estructura del Proyecto

El código está organizado en módulos separados para facilitar el mantenimiento:

```text
.
├── config/              # Configuración (Hydra YAML)
│   └── config.yaml      # Parámetros del modelo y datos
├── data/                # Dataset (German Credit Data)
├── images/              # Gráficos y visualizaciones
├── mlruns/              # Experimentos MLflow
├── notebooks/           # Análisis exploratorio e interpretabilidad
├── src/                 # Código fuente
│   ├── __init__.py
│   ├── api.py           # API REST (FastAPI)
│   ├── dashboard.py     # Dashboard interactivo (Streamlit)
│   ├── pipeline.py      # Pipeline de ML
│   ├── preprocessing.py # Limpieza de datos
│   └── train.py         # Entrenamiento con MLflow
├── .dockerignore        # Archivos excluidos de Docker
├── .gitignore           # Archivos excluidos de Git
├── docker-compose.yml   # Configuración de contenedores
├── Dockerfile           # Imagen de Docker
├── Makefile             # Comandos simplificados
├── pyproject.toml       # Dependencias del proyecto
├── uv.lock              # Versiones exactas de dependencias
└── README.md            # Documentación
```

---

## 🕹️ Automatización (Makefile)

El proyecto incluye comandos simplificados para facilitar su uso:

| Comando | Descripción |
| :--- | :--- |
| `make help` | Muestra todos los comandos disponibles |
| `make install` | Instala las dependencias del proyecto |
| `make api` | Inicia el servidor API en local |
| `make dashboard` | Inicia el dashboard interactivo |
| `make docker-build` | Construye las imágenes de Docker |
| `make docker-up` | Inicia todo el sistema con Docker |
| `make docker-down` | Detiene todos los contenedores |
| `make clean` | Limpia archivos temporales y caché |

---

## 💻 Instalación y Uso

### Opción A: Docker (Recomendada)

1. **Inicia el sistema completo:**

    ```bash
    make docker-up
    ```

    La primera vez puede tardar unos minutos mientras descarga las imágenes.

2. **Acceder a las interfaces:**

    * Dashboard: http://localhost:8501
    * API: http://localhost:8000/docs

3. **Detener el sistema:**

    ```bash
    make docker-down
    ```

### Opción B: Ejecución Local

Para desarrollo o si prefieres ejecutar sin Docker:

1. **Instalar dependencias:**

    ```bash
    make install
    ```

2. **Ejecutar servicios (en terminales separadas):**

    ```bash
    make api        # Terminal 1: Inicia la API
    make dashboard  # Terminal 2: Inicia el dashboard
    ```

3. **Acceder a las interfaces:**

    * Dashboard: http://localhost:8501
    * API: http://localhost:8000/docs

**Nota:** Asegúrate de tener el archivo `final_model.pkl` en la raíz del proyecto antes de ejecutar la API o el dashboard.

---

## 🧠 Dashboard & API

El sistema ofrece dos formas de interactuar con el modelo:

### 1. Dashboard Interactivo (Streamlit)

Interfaz web simple y visual:

* **Formulario de datos:** Campos para introducir el perfil del cliente
* **Predicción en tiempo real:** Muestra la probabilidad de impago
* **Explicabilidad con SHAP:** Gráficos que muestran qué variables (edad, saldo, historial crediticio) influyen más en la decisión

### 2. API REST (FastAPI)

Endpoint programático para integraciones:

* **Endpoint `/predict`:** Recibe el perfil del cliente en formato JSON y devuelve la predicción
* **Validación automática:** Verifica que los datos de entrada sean correctos
* **Documentación interactiva:** Interfaz Swagger en `/docs` para probar la API directamente desde el navegador

---

## ⚙️ Metodología de Data Science

### 1. Ingeniería de Datos:

* **Dataset:** German Credit Data con información de clientes bancarios
* **Limpieza:** Mapeo de variables categóricas codificadas (ej: A11 → "Saldo Negativo")
* **Preprocesamiento:** Pipeline con escalado para variables numéricas y codificación One-Hot para categóricas

### 2. Modelado:

* **Validación Cruzada Anidada:** Evita sobreajuste al seleccionar el mejor modelo
* **Modelos evaluados:** Logistic Regression, KNN, Random Forest, XGBoost
* **Optimización:** Búsqueda bayesiana de hiperparámetros con Optuna maximizando F1-Score
* **Tracking:** Todos los experimentos registrados en MLflow

### 3. Interpretabilidad:

* **SHAP Values:** Explica cada predicción mostrando qué variables son más importantes
* **Transparencia:** Permite entender por qué el modelo toma cada decisión

---

## 📊 Resultados

**XGBoost** fue seleccionado como modelo de producción por su excelente desempeño:

* **Manejo de desbalanceo de clases:** Penaliza correctamente los falsos negativos (clientes de alto riesgo)
* **Captura de relaciones no lineales:** Detecta patrones complejos entre variables
* **Robustez:** Rendimiento consistente en validación cruzada

![Comparación de modelos mediante Validación Cruzada Anidada](images/ncv_model_comparison.png)

Todos los experimentos están disponibles en MLflow con métricas, parámetros y artefactos versionados.

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
* **LinkedIn:** [Juan Pedro García Sanz](https://www.linkedin.com/in/juanpedrogarciasanz)
