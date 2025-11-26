import hydra
from omegaconf import DictConfig
import joblib
import pandas as pd

# Importamos nuestro módulo de preprocesamiento
from src.preprocessing import load_data, clean_dataframe, split_features_target
from src.pipeline import get_training_pipeline


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 Iniciando entrenamiento final para: {cfg.model.name}")
    
    # --- 1. INGENIERÍA DE DATOS ---
    df_raw = load_data(cfg.data.path)
    df_clean = clean_dataframe(df_raw)
    X, y = split_features_target(df_clean, cfg.data.target_col)



    # --- 2. DEFINICIÓN DEL PIPELINE ---
    # Identificamos columnas dinámicamente
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"🔧 Configurando Pipeline con parámetros del YAML...")

    # Convertimos la config de Hydra a diccionario normal de Python
    model_params = dict(cfg.model.params)
    
    pipeline = get_training_pipeline(model_params, num_cols, cat_cols)

    # --- 3. ENTRENAMIENTO ---
    print("Training model with full dataset...")
    pipeline.fit(X, y)

    print("✅ Entrenamiento completado.")
    
    # --- 4. GUARDADO DEL MODELO ---
    model_path = "final_model.pkl"
    joblib.dump(pipeline, model_path)
    
    print("-" * 60)
    print(f"💾 Modelo final guardado en: {model_path}")
    print("🚀 Listo para desplegar en Docker/API")
    print("-" * 60)

if __name__ == "__main__":
    main()