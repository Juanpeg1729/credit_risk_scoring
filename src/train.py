import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

# Importamos las funciones que acabas de crear en preprocessing.py
# Esto conecta tu script principal con tu módulo de limpieza
from src.preprocessing import load_data, clean_dataframe, split_features_target

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # 1. Mensaje de inicio (igual que antes, para saber qué configuración usas)
    print(f"🚀 Iniciando pipeline para el modelo: {cfg.model.name}")
    print(f"📂 Ruta de datos configurada: {cfg.data.path}")
    
    # 2. Carga de Datos (Usando tu nueva función)
    try:
        df_raw = load_data(cfg.data.path)
        print(f"📦 Datos crudos cargados. Dimensiones: {df_raw.shape}")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo csv. Revisa que esté en la carpeta 'data/'.")
        return

    # 3. Limpieza y Preprocesamiento (Usando tu nueva función)
    # Aquí es donde se arregla lo del '[5E-1]', se borran duplicados, etc.
    df_clean = clean_dataframe(df_raw)
    print(f"✨ Datos limpios. Dimensiones: {df_clean.shape}")

    # 4. Separación de Features (X) y Target (y)
    X, y = split_features_target(df_clean, cfg.data.target_col)

    # 5. División Train/Test 
    # Fíjate como usamos 'cfg.data.test_size' en vez de escribir 0.2 a mano.
    # ¡Eso es código profesional! Si quieres cambiarlo a 0.3, solo tocas el yaml.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=cfg.data.test_size, 
        random_state=cfg.data.random_state,
        stratify=y
    )
    
    print("-" * 30)
    print(f"📊 Conjunto de Entrenamiento (Train): {X_train.shape}")
    print(f"📊 Conjunto de Prueba (Test): {X_test.shape}")
    print("-" * 30)
    
    print("\n✅ Fase de Carga y Limpieza completada con éxito.")

if __name__ == "__main__":
    main()