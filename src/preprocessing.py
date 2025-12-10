import pandas as pd
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
    """Carga el dataset german.data sin cabeceras."""
    print(f"Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y prepara el dataframe para entrenamiento."""
    df = df.copy()

    # Eliminar duplicados
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Eliminadas {initial_rows - len(df)} filas duplicadas")
    
    # Normalizar target: 1 (Good) -> 0, 2 (Bad) -> 1
    if 'Risk' in df.columns:
        df['Risk'] = df['Risk'].map({1: 0, 2: 1})
        df = df.dropna(subset=['Risk'])
        
        risk_counts = df['Risk'].value_counts()
        print(f"Target normalizado - Distribución: {risk_counts.to_dict()}")

    # Convertir columnas categóricas a string
    categorical_cols = [
        "Status_of_checking_account", "Credit_history", "Purpose", 
        "Savings_account_bonds", "Present_employment_since", 
        "Personal_status_and_sex", "Other_debtors_guarantors", 
        "Property", "Other_installment_plans", "Housing", 
        "Telephone", "foreign_worker"
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Conversión monetaria DM -> EUR (factor 0.85)
    if 'Credit_amount' in df.columns:
        df['Credit_amount'] = (df['Credit_amount'] * 0.85).round(0).astype(int)

    return df


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features y target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y