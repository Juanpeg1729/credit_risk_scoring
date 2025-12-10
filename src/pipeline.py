from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from typing import Dict, List


def get_training_pipeline(model_params: Dict, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    """Construye el pipeline completo: preprocesamiento + modelo XGBoost."""
    
    # Pipeline numérico: imputación + escalado
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline categórico: imputación + OneHot
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinar ambos pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], verbose_feature_names_out=False)

    # Pipeline final: preprocesador + clasificador
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(**model_params)) # **model_params desempaqueta el diccionario del config.yaml
    ])

    return pipeline
