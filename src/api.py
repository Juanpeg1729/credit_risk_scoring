from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Definimos la estructura de datos (Input)
class CreditInput(BaseModel):
    Status_of_checking_account: str
    Duration_in_month: int
    Credit_history: str
    Purpose: str
    Credit_amount: int
    Savings_account_bonds: str
    Present_employment_since: str
    Installment_rate_in_percentage_of_disposable_income: int
    Personal_status_and_sex: str
    Other_debtors_guarantors: str
    Present_residence_since: int
    Property: str
    Age_in_years: int
    Other_installment_plans: str
    Housing: str
    Number_of_existing_credits_at_this_bank: int
    Job: str
    Number_of_people_being_liable_to_provide_maintenance_for: int
    Telephone: str
    foreign_worker: str

    # Un solo bloque de configuración para el ejemplo del Swagger
    class Config:
        json_schema_extra = {
            "example": {
                "Status_of_checking_account": "A11",
                "Duration_in_month": 6,
                "Credit_history": "A32",
                "Purpose": "A43",
                "Credit_amount": 1169,
                "Savings_account_bonds": "A65",
                "Present_employment_since": "A75",
                "Installment_rate_in_percentage_of_disposable_income": 4,
                "Personal_status_and_sex": "A93",
                "Other_debtors_guarantors": "A101",
                "Present_residence_since": 4,
                "Property": "A121",
                "Age_in_years": 67,
                "Other_installment_plans": "A143",
                "Housing": "A152",
                "Number_of_existing_credits_at_this_bank": 2,
                "Job": "A173",
                "Number_of_people_being_liable_to_provide_maintenance_for": 1,
                "Telephone": "A192",
                "foreign_worker": "A201"
            }
        }

# 2. Inicializamos la App
app = FastAPI(title="Credit Risk API")
model = None

# 3. Cargar el modelo al arrancar
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("final_model.pkl")

# 4. Endpoint de Predicción
@app.post("/predict")
def predict(data: CreditInput):
    input_df = pd.DataFrame([data.dict()])
    
    # Realizamos la predicción
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    
    # Retornamos el resultado JSON
    return {
        "prediction": "Risk" if pred == 1 else "No Risk",
        "probability": float(proba),
        "status": "Success"
    }