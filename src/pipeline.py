from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from typing import Dict, List

def get_training_pipeline(model_params: Dict, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """
    Construye el pipeline completo: Preprocesamiento + Modelo (XGBoost).
    """
    
    # 1. Pipeline Numérico: Imputación (por si acaso) + Escalado
    # Nota: La limpieza "dura" del [5E-1] ya la hicimos en preprocessing.py
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline Categórico: Imputación + OneHot
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. ColumnTransformer: Une ambos mundos
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], verbose_feature_names_out=False)

    # 4. El Modelo Final (XGBoost)
    # **model_params desempaqueta el diccionario del config.yaml (n_estimators, learning_rate, etc.)
    model = XGBClassifier(**model_params)

    # 5. Pipeline Final
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline
