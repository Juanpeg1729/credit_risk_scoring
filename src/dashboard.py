import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Adult Income Predictor", layout="wide")

st.title("💰 Predictor de Ingresos (con Interpretabilidad)")
st.markdown("Descubre si una persona gana más de **$50K/año** y *por qué*.")

# 2. Carga del Modelo (con caché para que sea rápido)
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("❌ No se encuentra el modelo. Ejecuta primero: uv run python -m src.train")
    st.stop()

# 3. Sidebar: Formulario de entrada de datos
st.sidebar.header("📝 Datos del Perfil")

def user_input_features():
    # ... (Los sliders y selectbox los dejas igual) ...
    age = st.sidebar.slider("Edad", 17, 90, 30)
    workclass = st.sidebar.selectbox("Clase de Trabajo", 
                ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    education_num = st.sidebar.slider("Años de Educación", 1, 16, 10)
    marital_status = st.sidebar.selectbox("Estado Civil", 
                ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.sidebar.selectbox("Ocupación", 
                ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.sidebar.selectbox("Relación", 
                ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.sidebar.selectbox("Raza", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.sidebar.selectbox("Sexo", ['Female', 'Male'])
    capital_gain = st.sidebar.number_input("Ganancia de Capital", 0, 100000, 0)
    capital_loss = st.sidebar.number_input("Pérdida de Capital", 0, 5000, 0)
    hours_per_week = st.sidebar.slider("Horas por semana", 1, 100, 40)
    native_country = st.sidebar.selectbox("País", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy', 'Dominican-Republic', 'Japan', 'Guatemala', 'Poland', 'Vietnam', 'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Nicaragua', 'Peru', 'Ecuador', 'France', 'Greece', 'Ireland'])

    # 👇 AQUÍ ESTÁ EL CAMBIO: Usamos puntos (.) en lugar de guiones (-)
    data = {
        'age': [age], 
        'workclass': [workclass], 
        'education.num': [education_num],       # Antes education-num
        'marital.status': [marital_status],     # Antes marital-status
        'occupation': [occupation],
        'relationship': [relationship], 
        'race': [race], 
        'sex': [sex],
        'capital.gain': [capital_gain],         # Antes capital-gain
        'capital.loss': [capital_loss],         # Antes capital-loss
        'hours.per.week': [hours_per_week],     # Antes hours-per-week
        'native.country': [native_country]      # Antes native-country
    }
    return pd.DataFrame(data)

input_df = user_input_features()

# 4. Panel Principal: Predicción
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Predicción")
    if st.button("Calcular"):
        # Predicción
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"**>50K USD** (Probabilidad: {proba:.2%})")
        else:
            st.warning(f"**<=50K USD** (Probabilidad: {proba:.2%})")

        # 5. SHAP: Explicación en tiempo real
        st.subheader("🧠 ¿Por qué?")
        with st.spinner("Calculando explicación SHAP..."):
            
            # a) Preparamos los datos (Transformación)
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier']
            
            X_trans = preprocessor.transform(input_df)
            
            # b) Recuperamos nombres de features (Plan B manual si falla automático)
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                num_cols = input_df.select_dtypes(include=['number']).columns.tolist()
                cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out()
                feature_names = num_cols + list(cat_names)

            # c) Calculamos SHAP (Usando el método robusto KernelExplainer)
            # Para hacerlo rápido en web, usamos una función lambda directa
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            
            # Usamos un fondo sintético pequeño para velocidad (dummy background)
            # En producción usaríamos un dataset resumen guardado, aquí lo creamos al vuelo con ceros para velocidad
            # Ojo: Esto es una aproximación para velocidad en la demo
            background = np.zeros((1, X_trans.shape[1])) 
            explainer = shap.KernelExplainer(predict_fn, background)
            
            shap_values = explainer.shap_values(X_trans)

            # d) Graficamos
            fig, ax = plt.subplots()
            # Waterfall plot es el mejor para explicar UNA sola predicción
            # Ajustamos formato para waterfall (espera objeto Explanation)
            explanation = shap.Explanation(
                values=shap_values[0], 
                base_values=explainer.expected_value, 
                data=X_trans[0],  # <--- CAMBIO CLAVE: Usamos el dato transformado
                feature_names=feature_names
            )
            
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)

with col2:
    st.info("👈 Modifica los parámetros en la barra lateral para ver cómo cambia la predicción.")