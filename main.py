import sys
import joblib
import pandas as pd
import numpy as np
import os
from fastapi import FastAPI, HTTPException, Body
from sklearn.base import BaseEstimator, TransformerMixin

# 1. MUST include your custom classes for joblib to load properly
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): pass
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_engineered = X.copy()
        if 'Post_Semester_GPA' in X_engineered.columns and 'Pre_Semester_GPA' in X_engineered.columns:
            X_engineered['GPA_Difference'] = X_engineered['Post_Semester_GPA'] - X_engineered['Pre_Semester_GPA']
            X_engineered = X_engineered.drop(columns=['Pre_Semester_GPA', 'Post_Semester_GPA'], errors='ignore')
        if 'Traditional_Study_Hours' in X_engineered.columns and 'Weekly_GenAI_Hours' in X_engineered.columns:
            X_engineered['Study_Balance'] = X_engineered['Traditional_Study_Hours'] - X_engineered['Weekly_GenAI_Hours']
            X_engineered = X_engineered.drop(columns=['Traditional_Study_Hours', 'Weekly_GenAI_Hours'], errors='ignore')
        if 'Student_ID' in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=['Student_ID'], errors='ignore')
        return X_engineered

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, lower_percentile=0.01, upper_percentile=0.99):
        self.features = features
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.capping_values_ = {}
    def fit(self, X, y=None):
        if self.features is None: self.features = X.select_dtypes(include=['number']).columns.tolist()
        for col in self.features:
            lower_bound = X[col].quantile(self.lower_percentile)
            upper_bound = X[col].quantile(self.upper_percentile)
            self.capping_values_[col] = {'lower': lower_bound, 'upper': upper_bound}
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col, bounds in self.capping_values_.items():
            X_transformed[col] = X_transformed[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        return X_transformed

# Register classes so joblib can find them
sys.modules['__main__'].FeatureEngineer = FeatureEngineer
sys.modules['__main__'].OutlierCapper = OutlierCapper

app = FastAPI(title="STUDENT BURNOUT PREDICTION API", version="1.4.3")

# Load artifacts
# Path relative to app/main.py
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "burnout_prediction_model_artifacts.joblib")

try:
    pipeline = joblib.load(ARTIFACT_PATH)
    # If your artifact is just the pipeline, 'pipeline' is the object. 
    # If it was a dict, access it: pipeline = artifacts['full_pipeline']
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Load failed: {e}")
    pipeline = None

@app.post("/predict")
def predict_burnout(student_data: dict = Body(...)):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        # Convert input to DataFrame
        df_input = pd.DataFrame([student_data])
        
        # Everything happens here: Imputation, Scaling, Encoding, RFE, and Regression
        raw_score = float(pipeline.predict(df_input)[0])
        
        # (Assuming you still want risk levels)
        # You can store thresholds in your artifact dict if needed
        return {
            "status": "success",
            "raw_regression_score": round(raw_score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))