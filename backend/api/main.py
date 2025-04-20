# backend/main.py

from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat, constr
import pickle
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# --- Load the Trained Model ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("Error loading the model:", e)

# --- Define Input Schema ---
class CustomerInput(BaseModel):
    tenure:        conint(ge=0)
    monthly_charges: confloat(ge=0.0)
    contract:      constr(strict=True)
    gender:        constr(strict=True)
    senior_citizen: conint(ge=0, le=1)
    partner:       constr(strict=True)
    dependents:    constr(strict=True)

    internet_service:  constr(strict=True)
    paperless_billing: constr(strict=True)
    payment_method:    constr(strict=True)

    # ONE field instead of seven: a list of service names
    services_list:    List[constr(strict=True)] = []

def make_feature_df(data: CustomerInput) -> pd.DataFrame:
    d = data.dict()

    # 1) Derive total_charges & avg_charge_per_month
    d['total_charges'] = d['tenure'] * d['monthly_charges']
    d['avg_charge_per_month'] = d['total_charges'] / (d['tenure'] + 1e-6)

    # 2) Tenure bucket
    t = d['tenure']
    if t <= 12:
        d['tenure_bucket'] = '0-12'
    elif t <= 24:
        d['tenure_bucket'] = '12-24'
    elif t <= 48:
        d['tenure_bucket'] = '24-48'
    else:
        d['tenure_bucket'] = '48+'

    # 3) Reconstruct the 7 service flags from services_list
    svc_map = {
        "Multiple Lines":     "multiple_lines",
        "Online Security":    "online_security",
        "Online Backup":      "online_backup",
        "Device Protection":  "device_protection",
        "Tech Support":       "tech_support",
        "Streaming TV":       "streaming_tv",
        "Streaming Movies":   "streaming_movies",
    }
    # initialize all to "No"
    for col in svc_map.values():
        d[col] = "No"
    # mark selected as "Yes"
    for svc in d.pop('services_list', []):
        key = svc_map.get(svc)
        if key:
            d[key] = "Yes"

    # 4) Compute services_count if your pipeline uses it
    d['services_count'] = sum(1 for col in svc_map.values() if d[col] == "Yes")

    # 5) Build DataFrame with exact column order your pipeline expects
    cols = [
        'tenure', 'monthly_charges', 'total_charges', 'avg_charge_per_month',
        'services_count', 'contract', 'gender', 'senior_citizen', 'partner',
        'dependents', 'internet_service', 'paperless_billing', 'payment_method',
        # the 7 flags:
        'multiple_lines', 'online_security', 'online_backup',
        'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
        'tenure_bucket'
    ]
    return pd.DataFrame([d], columns=cols)

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_churn(data: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = make_feature_df(data)
        proba = model.predict_proba(df)[0, 1]
        threshold = 0.5  # or your tuned threshold
        prediction = "Yes" if proba >= threshold else "No"
        return {"prediction": prediction, "probability": proba}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Health Check ---
@app.get("/health")
async def health():
    return {"status": "ok"}
