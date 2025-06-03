import joblib
import numpy as np
import os
from django.conf import settings
import pandas as pd

model_path = os.path.join(settings.BASE_DIR, 'analysis', 'model', 'model.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'analysis', 'model', 'scaler.pkl')


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]

def predict_diabetes(data):
    input_df = pd.DataFrame([data], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0].tolist() if hasattr(model, "predict_proba") else None
    
    return {
        "prediction": int(prediction),
        "result": "Diabetic" if prediction == 1 else "Non-Diabetic",
    }



def get_feature_importance():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))
    return {"message": "Feature importance not available for this model."}