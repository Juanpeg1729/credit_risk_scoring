# 🎯 Adult Income Predcition (End-to-End MLOps Pipeline)

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-ready-009688)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

Este proyecto desarrolla un flujo de trabajo completo de Machine Learning para predecir si una persona gana más de $50,000 anuales basándose en datos demográficos y laborales.

El enfoque principal de este repositorio es demostrar un **flujo de trabajo de machine learning robusto y profesional**, desde la limpieza de datos hasta la selección de modelos mediante **Validación Cruzada Anidada (NCV)** para obtener una estimación de rendimiento imparcial.

---

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Características Principales](#características-principales)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación y Uso](#instalación-y-uso)
- [Metodología](#metodología)
- [Resultados](#resultados)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Autor](#autor)

---

## 🧐 Descripción del Proyecto

Utilizando el famoso dataset **"Adult Census Income"** del repositorio [UCI Machine Learning](https://archive.ics.uci.edu/dataset/2/adult), este proyecto aborda un problema de clasificación binaria desbalanceada.

El objetivo no es solo obtener la mayor precisión, sino construir un **pipeline reproducible y profesional** que incluya limpieza de datos, ingeniería de características, selección de modelos imparcial y ajuste de hiperparámetros.

---

## 🚀 Características Principales

* **Preprocesamiento Robusto:** Uso de `ColumnTransformer` y `Pipeline` de Scikit-Learn para encapsular la limpieza, imputación de nulos y codificación (OneHotEncoding) de variables categóricas.
* **Validación Cruzada Anidada (NCV):** Implementación de una estrategia de 5 folds exteriores y 3 interiores para separar la optimización de hiperparámetros de la evaluación del error, garantizando resultados realistas.
* **Manejo de Desbalanceo:** Configuración específica de pesos de clase (`scale_pos_weight`) en modelos de boosting.
* **Comparativa de Modelos:** Evaluación de Regresión Logística, KNN, Random Forest y XGBoost.
* **Gestión de Dependencias Moderna:** Uso de `pyproject.toml` para una instalación limpia y estandarizada.
* **Reproducibilidad:** Control estricto de la aleatoriedad mediante semillas globales (`SEED`).

---

## 🛠 Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Entorno:** Jupyter Notebook / Google Colab

---

## 💻 Instalación y Uso

Este proyecto utiliza `uv`, un instalador y gestor de entornos virtuales de Python de alto rendimiento, para una configuración rápida. Las dependencias están definidas en el archivo `pyproject.toml`.

**Prerrequisito:** Asegúrate de tener `uv` instalado. Si no es así, consulta la [guía oficial de instalación de UV](https
://astral.sh/uv#installation).

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/](https://github.com/)[TU_USUARIO]/adult-income-analysis.git
    cd adult-income-analysis
    ```

2.  **Crear el entorno virtual:**
    `uv` creará un entorno virtual llamado `.venv` en el directorio actual.
    ```bash
    uv venv
    ```

3.  **Activar el entorno:**
    ```bash
    # macOS / Linux
    source .venv/bin/activate
    
    # Windows (PowerShell)
    .venv\Scripts\Activate.ps1
    
    # Windows (CMD)
    .venv\Scripts\activate.bat
    ```

4.  **Instalar dependencias:**
    `uv` leerá el archivo `pyproject.toml` e instalará todas las dependencias del proyecto (incluyendo `jupyter`) a gran velocidad.
    ```bash
    uv pip install .
    ```

5.  **Ejecutar el Notebook:**
    ```bash
    jupyter notebook notebooks/Proyecto_Adult_Income.ipynb
    ```

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
