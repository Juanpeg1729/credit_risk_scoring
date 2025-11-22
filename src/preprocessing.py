import pandas as pd
import numpy as np
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    print(f"   csv cargando desde: {filepath}")
    return pd.read_csv(filepath)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza inicial (Pandas) al dataframe:
    - Elimina columnas redundantes.
    - Elimina duplicados.
    - Convierte target a binario.
    - Sanea valores extraños como '[5E-1]'.
    """
    df = df.copy()
    
    # 1. Eliminación de columnas (Lógica de tu notebook)
    # fnlwgt: estadística irrelevante para el modelo
    # education: redundante con education-num
    cols_to_drop = ['fnlwgt', 'education']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # 2. Eliminación de duplicados
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    print(f"   🧹 Filas duplicadas eliminadas: {original_len - len(df)}")

    # 3. Saneamiento de valores '?' (Los convertimos a NaN para que SimpleImputer los maneje luego)
    df.replace('?', np.nan, inplace=True)
    
    # 4. Saneamiento CRÍTICO de datos sucios (El error [5E-1])
    # Forzamos que las columnas numéricas no tengan texto raro
    # Si encontramos algo que no es número en una columna numérica, lo hacemos NaN
    numeric_candidates = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for col in numeric_candidates:
        if col in df.columns:
            # to_numeric con coerce convierte errores (como [5E-1]) en NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. Transformación del Target (Variable Objetivo)
    if 'income' in df.columns:
        # Limpiamos espacios y convertimos a 0/1
        df['income'] = df['income'].astype(str).str.strip()
        df['income'] = (df['income'] == '>50K').astype(int)
        
    return df

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y