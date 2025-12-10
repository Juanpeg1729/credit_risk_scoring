import hydra
from omegaconf import DictConfig
import joblib
from pathlib import Path
import mlflow
from mlflow.models.signature import infer_signature

from src.preprocessing import load_data, clean_dataframe, split_features_target
from src.pipeline import get_training_pipeline

# El decorador @hydra.main se encarga de cargar la configuración desde config/config.yaml
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print(f"🚀 Iniciando entrenamiento: {cfg.model.name}")
    
    root_dir = Path(__file__).resolve().parent.parent # Este es el directorio raíz del proyecto
    
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file://{root_dir}/mlruns") # Ruta local para almacenar los experimentos
    exp_name = cfg.mlflow.get("experiment_name", "Credit_Risk_Default") # Nombre del experimento. Por defecto "Credit_Risk_Default"
    mlflow.set_experiment(exp_name) # Establecer experimento en MLflow

    # Cargar y preparar datos
    try:
        data_path = root_dir / cfg.data.path 
        df_raw = load_data(str(data_path)) # str porque load_data espera un string
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    df_clean = clean_dataframe(df_raw)
    X, y = split_features_target(df_clean, cfg.data.target_col)

    # Crear pipeline
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    model_params = dict(cfg.model.params)
    pipeline = get_training_pipeline(model_params, num_cols, cat_cols)

    # Entrenar y registrar en MLflow
    with mlflow.start_run() as run:
        print(f"🔧 Run ID: {run.info.run_id}")
        
        mlflow.log_params(model_params)
        mlflow.log_param("data_source", str(cfg.data.path))
        
        print("Entrenando modelo...")
        pipeline.fit(X, y)
        
        # Guardar modelo en MLflow con signature
        input_example = X.iloc[:5]
        signature = infer_signature(X, pipeline.predict(X))
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        
        # Guardar modelo local para API/Dashboard
        save_path = root_dir / "final_model.pkl"
        joblib.dump(pipeline, save_path)
        
        print(f"✅ Modelo guardado en: {save_path}")
        print(f"✅ Experimento '{exp_name}' registrado en MLflow")

if __name__ == "__main__":
    main()