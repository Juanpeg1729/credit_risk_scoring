from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# 1. Definimos la estructura de los datos de entrada
# Usamos Field para dar ejemplos en la documentación automática
class IncomeInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State.gov")
    education_num: int = Field(..., alias="education.num", example=13) # alias maneja el guion
    marital_status: str = Field(..., alias="marital.status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital.gain", example=2174)
    capital_loss: int = Field(..., alias="capital.loss", example=0)
    hours_per_week: int = Field(..., alias="hours.per.week", example=40)
    native_country: str = Field(..., alias="native.country", example="United-States")

    class Config:
        populate_by_name = True # Permite usar education_num o education-num

# 2. Inicializamos la APP
app = FastAPI(title="Adult Income Prediction API", version="1.0.0")

# 3. Variable global para el modelo
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "final_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ No se encuentra el modelo. Ejecuta 'uv run python -m src.train' primero.")
    model = joblib.load(model_path)
    print("✅ Modelo cargado en memoria")

@app.post("/predict")
def predict_income(input_data: IncomeInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    # Convertimos el Pydantic model a DataFrame
    # by_alias=True es CRÍTICO para recuperar los nombres con guiones (education-num)
    data_dict = input_data.model_dump(by_alias=True)
    df_input = pd.DataFrame([data_dict])
    
    # Hacemos la predicción
    try:
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1] # Probabilidad de la clase 1 (>50K)
        
        result = ">50K" if prediction == 1 else "<=50K"
        
        return {
            "prediction": result,
            "probability_high_income": round(float(probability), 4),
            "message": "Predicción exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/")
def root():
    return {"message": "API de Predicción de Ingresos funcionando. Ve a /docs para probarla."}