import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Scoring Riesgo Crédito", layout="wide", page_icon="🏦")

st.title("🏦 Scoring de Riesgo Crediticio")
st.markdown("""
Esta herramienta evalúa la probabilidad de **impago** de un cliente basándose en su perfil financiero.
Utiliza un modelo **XGBoost** entrenado con datos del mercado alemán. Selecciona las características del solicitante en la barra lateral y pulsa **Calcular Riesgo** para obtener el resultado.
""")

# 2. Carga del Modelo
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("❌ No se encuentra el modelo. Ejecuta primero: make train")
    st.stop()

# 3. Sidebar: Formulario
st.sidebar.header("📝 Perfil del Solicitante")

def user_input_features():
    # --- VARIABLES NUMÉRICAS ---
    duration = st.sidebar.slider("Duración (meses)", 4, 72, 24)
    amount = st.sidebar.number_input("Monto del Crédito (€)", 200, 20000, 2000)
    age = st.sidebar.slider("Edad", 19, 75, 30)
    
    # --- VARIABLES CATEGÓRICAS (Mapeos Limpios) ---
    
    # Estado Cuenta
    map_checking = {
        "En negativo (< 0 €)": "A11",
        "Saldo mínimo (0 - 200 €)": "A12",
        "Saldo positivo (> 200 €)": "A13",
        "Sin cuenta corriente": "A14"
    }
    selected_checking = st.sidebar.selectbox("Estado Cuenta Corriente", options=list(map_checking.keys()))
    checking_code = map_checking[selected_checking]

    # Historial
    map_history = {
        "Sin créditos / Todos pagados": "A30",
        "Todos pagados (este banco)": "A31",
        "Pagos al día (vigentes)": "A32",
        "Retrasos en el pasado": "A33",
        "Cuenta crítica / Otros créditos": "A34"
    }
    selected_history = st.sidebar.selectbox("Historial de Crédito", options=list(map_history.keys()))
    history_code = map_history[selected_history]

    # Propósito
    map_purpose = {
        "Coche (Nuevo)": "A40",
        "Coche (Usado)": "A41",
        "Mobiliario / Equipamiento": "A42",
        "Radio / TV": "A43",
        "Electrodomésticos": "A44",
        "Reparaciones": "A45",
        "Educación": "A46",
        "Negocios": "A49",
        "Otros": "A410"
    }
    selected_purpose = st.sidebar.selectbox("Propósito", options=list(map_purpose.keys()))
    purpose_code = map_purpose[selected_purpose]

    # Esfuerzo de Pago
    map_installment = {
        "Muy cómodo (< 20% de la renta)": 1,
        "Cómodo (20% - 25% de la renta)": 2,
        "Ajustado (25% - 35% de la renta)": 3,
        "Crítico (> 35% de la renta)": 4
    }
    selected_installment = st.sidebar.selectbox("Esfuerzo de Pago (% de Ingresos)", options=list(map_installment.keys()))
    installment_code = map_installment[selected_installment]
    
    # --- Construcción del DataFrame ---
    # Rellenamos las variables secundarias con valores por defecto (moda)
    data = {
        'Duration_in_month': [duration],
        'Credit_amount': [amount],
        'Age_in_years': [age],
        'Installment_rate_in_percentage_of_disposable_income': [installment_code],
        'Status_of_checking_account': [checking_code],
        'Credit_history': [history_code],
        'Purpose': [purpose_code],
        
        # Valores por defecto para variables no mostradas (Simplificación UX)
        'Savings_account_bonds': ['A61'], 
        'Present_employment_since': ['A73'],
        'Personal_status_and_sex': ['A93'],
        'Other_debtors_guarantors': ['A101'],
        'Present_residence_since': [2],
        'Property': ['A121'],
        'Other_installment_plans': ['A143'], 
        'Housing': ['A152'], 
        'Number_of_existing_credits_at_this_bank': [1],
        'Job': ['A173'], 
        'Number_of_people_being_liable_to_provide_maintenance_for': [1],
        'Telephone': ['A191'], 
        'foreign_worker': ['A201'] 
    }
    
    return pd.DataFrame(data)

input_df = user_input_features()

# 4. Panel Principal
# Dividimos en 2 columnas: Izquierda (Resultado), Derecha (Gráfico)
col1, col2 = st.columns([1, 2])

# Botón único para calcular todo
if st.sidebar.button("Calcular Riesgo", type="primary"):
    
    # --- LÓGICA DE PREDICCIÓN ---
    with col1:
        st.subheader("🔍 Resultado")
        
        # Predicción
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1] # Probabilidad de Riesgo (Clase 1)

        if prediction == 1:
            st.error(f"🔴 **ALTO RIESGO**")
            st.metric(label="Probabilidad de Impago", value=f"{proba:.2%}")
            st.markdown("---")
            st.markdown("**Recomendación:** Denegar la operación o solicitar avales adicionales.")
        else:
            st.success(f"🟢 **BAJO RIESGO**")
            st.metric(label="Probabilidad de Impago", value=f"{proba:.2%}")
            st.markdown("---")
            st.markdown("**Recomendación:** Conceder el crédito.")

    # --- LÓGICA DE EXPLICABILIDAD (SHAP) ---
    with col2:
        st.subheader("🧠 Factores Clave")
        with st.spinner("Analizando motivos..."):
            
            # 1. Preparar datos
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['classifier']
            X_trans = preprocessor.transform(input_df)
            
            # 2. Recuperar Nombres Crudos
            try:
                raw_feature_names = preprocessor.get_feature_names_out()
            except:
                num_cols = input_df.select_dtypes(include=['number']).columns.tolist()
                cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out()
                raw_feature_names = num_cols + list(cat_names)

            # 3. TRADUCCIÓN MAESTRA (Aquí está la magia) ✨
            # Diccionario de códigos alemanes -> Español
            code_map = {
                # Ahorros (Savings)
                "A61": "< 100 €", "A62": "100-500 €", "A63": "500-1000 €", "A64": "> 1000 €", "A65": "Desconocido",
                # Empleo (Employment)
                "A71": "Desempleado", "A72": "< 1 año", "A73": "1-4 años", "A74": "4-7 años", "A75": "> 7 años",
                # Vivienda (Housing)
                "A151": "Alquiler", "A152": "Propia", "A153": "Gratis",
                # Estado Civil/Sexo (Personal Status)
                "A91": "H. Divorciado", "A92": "M. Div/Casada", "A93": "H. Soltero", "A94": "H. Casado",
                # Trabajo (Job)
                "A171": "Desempleado", "A172": "No cualificado", "A173": "Cualificado", "A174": "Directivo",
                # Otros
                "A191": "Sin Tlf", "A192": "Con Tlf",
                "A201": "Extranjero", "A202": "Local",
                # (Añade aquí los códigos de Checking, History, etc. que ya tenías)
                "A11": "En Rojo", "A12": "Saldo Bajo", "A13": "Saldo Positivo", "A14": "Sin Cuenta",
                "A30": "Sin Créditos", "A31": "Pagados", "A32": "Al día", "A33": "Retrasos", "A34": "Crítico"
            }

            clean_names = []
            for name in raw_feature_names:
                # 1. Limpieza de prefijos técnicos del Pipeline
                new_name = name.replace("cat__", "").replace("num__", "")
                
                # 2. Limpieza de nombres de variables (Del inglés al español)
                new_name = new_name.replace("Savings_account_bonds_", "Ahorros: ")
                new_name = new_name.replace("Status_of_checking_account_", "Cuenta: ")
                new_name = new_name.replace("Duration_in_month", "Duración")
                new_name = new_name.replace("Credit_history_", "Historial: ")
                new_name = new_name.replace("Credit_amount", "Monto")
                new_name = new_name.replace("Age_in_years", "Edad")
                new_name = new_name.replace("Installment_rate_in_percentage_of_disposable_income", "Esfuerzo")
                new_name = new_name.replace("Present_employment_since_", "Empleo: ")
                new_name = new_name.replace("Personal_status_and_sex_", "Estado: ")
                new_name = new_name.replace("Housing_", "Vivienda: ")
                new_name = new_name.replace("Job_", "Trabajo: ")
                new_name = new_name.replace("foreign_worker_", "Origen: ")
                
                # 3. Sustitución de CÓDIGOS por TEXTO (Usando el diccionario)
                # Iteramos sobre el mapa para reemplazar cualquier código Axx que quede
                for code, text in code_map.items():
                    if code in new_name:
                        new_name = new_name.replace(code, text)
                
                clean_names.append(new_name)

            # 4. Calcular SHAP
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
            background = np.zeros((1, X_trans.shape[1])) 
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_trans)

            # 5. Graficar
            fig, ax = plt.subplots(figsize=(8, 6))
            explanation = shap.Explanation(
                values=shap_values[0], 
                base_values=explainer.expected_value, 
                data=X_trans[0], 
                feature_names=clean_names  # <--- Nombres limpios y traducidos
            )
            shap.plots.waterfall(explanation, max_display=8, show=False)
            st.pyplot(fig)