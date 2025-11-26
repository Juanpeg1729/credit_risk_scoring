# 🏦 Credit Risk Scoring: End-to-End MLOps Pipeline

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
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
* **Interfaces:** 
    * **FastAPI:** API REST para inferencia máquina-a-máquina.
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

Para facilitar el uso, el proyecto incluye un `Makefile` que abstrae los comandos complejos.

| Comando | Descripción |
| :--- | :--- |
| `make install` | Instala las dependencias con `uv`. |
| `make train` | Ejecuta el pipeline de entrenamiento completo. |
| `make api` | Levanta el servidor de la API (FastAPI) en local. |
| `make dashboard` | Lanza la aplicación web (Streamlit). |
| `make docker-up` | Levanta todo el sistema (API + Dashboard) en contenedores. |
| `make docker-run` | Ejecuta el contenedor con la App completa. |

---

## 💻 Instalación y Uso

Tienes dos formas de ejecutar este proyecto: la Profesional (Docker) y la de Desarrollo (Local).

**Opción A: Docker (Experiencia Completa)**
Ejecuta todo el sistema (Backend y Frontend) en contenedores aislados.

1.  **Construir y Arrancar:**
    Puedes usar el atajo con Make o el comando nativo de Docker.
    ```bash
    make docker-up
    # O si no tienes Make instalado:
    # docker-compose up --build
    ```

2.  **Acceder:**
    * 🎨 **Dashboard (Streamlit):** http://localhost:8501
    * ⚙️ **API (Swagger UI):** http://localhost:8000/docs

3.  **Apagar:**
    Pulsa `Ctrl+C` o ejecuta `make docker-down`.


**Opción B: Ejecución Local (Desarrollo)**

Si deseas editar el código o entrenar manualmente. Requiere tener uv y make instalados.

1. **Instalar dependencias:**

```bash
make install
```

2. **Entrenar el modelo (Genera final_model.pkl):**

```bash
make train
```

3. **Ejecutar interfaces:**

- Para API:
```
make api
```

- Para el dashboard:
```
make dashboard
```

---

## 🧠 Dashboard & Interpretabilidad (XAI)

El proyecto incluye un Dashboard interactivo construido con Streamlit que permite:

1. Simular perfiles de clientes mediante un formulario intuitivo.

2. Obtener la predicción de riesgo en tiempo real.

3. Entender el "Por qué": Integración de SHAP (SHapley Additive exPlanations) para visualizar qué variables específicas (edad, historial, saldo) empujaron la decisión del modelo hacia "Riesgo" o "Aprobado".

---

## ⚙️ Metodología de ML

Aunque el código ahora es modular, la lógica de Machine Learning subyacente se mantiene sólida:

1. **Ingeniería de Datos:** Ingestión de datos crudos (.data), mapeo de variables categóricas cifradas (ej: A11 -> Saldo Negativo) y normalización de moneda.

2. **Pipeline de Preprocesamiento:** ColumnTransformer para escalado numérico y codificación One-Hot, integrado en un Pipeline de Scikit-Learn.

3. **Selección de Modelos:** Comparativa mediante Validación Cruzada Anidada (Nested CV) para evitar el sobreajuste.

4. **Optimización:** Búsqueda bayesiana con Optuna para maximizar el F1-Score (dado el desbalanceo de clases).

---

## 📊 Resultados del Modelo

Tras la evaluación rigurosa, **XGBoost** fue seleccionado como el modelo de producción por su capacidad para manejar desbalanceo y relaciones no lineales. Los resultados fueron los siguientes:

![Comparación de modelos mediante Validación Cruzada Anidada](images/ncv_model_comparison.png)

---

## ✒️ Autor

**Juan Pedro García Sanz**

* **GitHub:** [@Juanpeg1729](https://github.com/Juanpeg1729)
* **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/juan-pedro-garcía-sanz-443b31343)
