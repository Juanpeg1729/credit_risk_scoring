import pandas as pd
import numpy as np
from typing import Tuple

# Nombres oficiales según la documentación de UCI para 'german.data'
COLUMN_NAMES = [
    "Status_of_checking_account", 
    "Duration_in_month", 
    "Credit_history", 
    "Purpose", 
    "Credit_amount", 
    "Savings_account_bonds", 
    "Present_employment_since", 
    "Installment_rate_in_percentage_of_disposable_income", 
    "Personal_status_and_sex", 
    "Other_debtors_guarantors", 
    "Present_residence_since", 
    "Property", 
    "Age_in_years", 
    "Other_installment_plans", 
    "Housing", 
    "Number_of_existing_credits_at_this_bank", 
    "Job", 
    "Number_of_people_being_liable_to_provide_maintenance_for", 
    "Telephone", 
    "foreign_worker", 
    "Risk"
]

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset 'german.data'.
    Al ser un archivo .data sin cabeceras y separado por espacios,
    necesitamos parámetros especiales.
    """
    print(f"   📂 Cargando datos crudos desde: {filepath}")
    
    # sep='\s+' significa "cualquier espacio en blanco (espacio o tabulador)"
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        header=None, 
        names=COLUMN_NAMES
    )
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y adapta el dataframe para el entrenamiento.
    """
    df = df.copy()
    
    # 1. Tratamiento del Target (Risk)
    # En el dataset original de UCI:
    #   1 = Good (Bueno)
    #   2 = Bad (Malo / Riesgo)
    #
    # Para Machine Learning (Detección de Riesgo/Fraude), la convención es:
    #   0 = Clase Negativa (Normal/Good)
    #   1 = Clase Positiva (Lo que buscamos/Bad)
    
    if 'Risk' in df.columns:
        # Mapeamos: 1 -> 0, 2 -> 1
        df['Risk'] = df['Risk'].map({1: 0, 2: 1})
        print(f"   🎯 Target 'Risk' normalizado: 1 (Bad) / 0 (Good)")
        
        # Validación rápida
        risk_counts = df['Risk'].value_counts()
        print(f"      Distribución: {risk_counts.to_dict()}")

    # 2. No necesitamos borrar columnas específicas como en Adult Income
    # porque este dataset es más técnico y todas las variables aportan valor.
    
    # 3. Tratamiento de tipos (Opcional pero recomendado)
    # Algunas columnas categóricas como 'Job' a veces vienen como números.
    # Es mejor forzarlas a texto para que el OneHotEncoder las trate bien.
    categorical_cols = ["Status_of_checking_account", "Credit_history", "Purpose", 
                        "Savings_account_bonds", "Present_employment_since", 
                        "Personal_status_and_sex", "Other_debtors_guarantors", 
                        "Property", "Other_installment_plans", "Housing", 
                        "Telephone", "foreign_worker"]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # NUEVO: Conversión monetaria a Euros actuales
    # Factor 0.85 aprox (Cambio DM->EUR + Inflación 30 años)
    if 'Credit_amount' in df.columns:
        df['Credit_amount'] = (df['Credit_amount'] * 0.85).round(0).astype(int)

    return df

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y