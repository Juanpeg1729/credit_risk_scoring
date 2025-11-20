import hydra
from omegaconf import DictConfig
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Importamos nuestro módulo de preprocesamiento
from src.preprocessing import load_data, clean_dataframe, split_features_target
from src.pipeline import get_training_pipeline


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 Iniciando entrenamiento profesional para: {cfg.model.name}")
    
    # --- 1. INGENIERÍA DE DATOS ---
    df_raw = load_data(cfg.data.path)
    df_clean = clean_dataframe(df_raw)
    X, y = split_features_target(df_clean, cfg.data.target_col)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state,
        stratify=y
    )
    print(f"✅ Datos listos. Train shape: {X_train.shape}")

    # --- 2. DEFINICIÓN DEL PIPELINE ---
    # Identificamos columnas dinámicamente
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"🔧 Configurando Pipeline con parámetros del YAML...")
    # Convertimos la config de Hydra a diccionario normal de Python
    model_params = dict(cfg.model.params)
    
    pipeline = get_training_pipeline(model_params, num_cols, cat_cols)

    # --- 3. ENTRENAMIENTO ---
    print("Training model... (Esto puede tardar un poco)")
    pipeline.fit(X_train, y_train)

    # --- 4. EVALUACIÓN ---
    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    print("-" * 60)
    print(f"🏆 Resultado Final (F1-Score): {f1:.4f}")
    print("-" * 60)
    print("Reporte detallado:")
    print(classification_report(y_test, y_pred))

    # --- 5. GUARDADO DEL MODELO (SERIALIZACIÓN) ---
    model_path = "final_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"💾 Modelo guardado exitosamente en: {model_path}")

if __name__ == "__main__":
    main()