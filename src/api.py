from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os


class CreditInput(BaseModel):
    Duration_in_month: int = Field(..., example=24)
    Credit_amount: int = Field(..., example=2100)
    Installment_rate_in_percentage_of_disposable_income: int = Field(..., example=4)
    Present_residence_since: int = Field(..., example=2)
    Age_in_years: int = Field(..., example=35)
    Number_of_existing_credits_at_this_bank: int = Field(..., example=1)
    Number_of_people_being_liable_to_provide_maintenance_for: int = Field(..., example=1)
    Status_of_checking_account: str = Field(..., example="A11")
    Credit_history: str = Field(..., example="A32")
    Purpose: str = Field(..., example="A43")
    Savings_account_bonds: str = Field(..., example="A61")
    Present_employment_since: str = Field(..., example="A73")
    Personal_status_and_sex: str = Field(..., example="A93")
    Other_debtors_guarantors: str = Field(..., example="A101")
    Property: str = Field(..., example="A121")
    Other_installment_plans: str = Field(..., example="A143")
    Housing: str = Field(..., example="A152")
    Job: str = Field(..., example="A173")
    Telephone: str = Field(..., example="A192")
    foreign_worker: str = Field(..., example="A201")

    class Config:
        populate_by_name = True


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

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
def predict_credit_risk(input_data: CreditInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    input_dict = input_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    result = "ALTO RIESGO (No conceder)" if prediction == 1 else "BAJO RIESGO (Conceder)"
    
    return {
        "prediction": result,
        "probability_default": round(float(probability), 4),
        "message": "Evaluación de Riesgo completada"
    }


@app.get("/")
def root():
    return {"message": "API de Credit Risk Scoring funcionando. Ve a /docs para probarla."}