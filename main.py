from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd

# Global variable placeholder for the model
model = None

# 1. Define modern Lifespan handler for model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load("student_performance_lgbm_model.pkl")
    except Exception as e:
        raise RuntimeError(f"Could not load the model file. Error: {str(e)}")
    yield
    # Clean up operations can go here if needed on shutdown

# 2. Initialize the FastAPI Application with lifespan
app = FastAPI(
    title="Student Performance Prediction API",
    description="A production-ready API to predict high-performing student outcomes using an optimized LightGBM model.",
    version="1.0",
    root_path="/proxy",
    lifespan=lifespan
)

# 3. Define the Input Data Schema using clean Pydantic V2 syntax
class StudentDataInput(BaseModel):
    Study_Balance: float = Field(..., description="Calculated student study balance metric", examples=[0.75])
    GPA_Difference: float = Field(..., description="Difference margin in student GPA scores", examples=[0.32])
    Skill_Retention_Score: float = Field(..., description="Testing retention score metric", examples=[0.81])
    Anxiety_Level_During_Exams: float = Field(..., description="Normalized metric for exam anxiety", examples=[0.45])
    Tool_Diversity: float = Field(..., description="Score metric representing tech tool diversity utilized", examples=[0.60])

# 4. Create the Home Route
@app.get("/")
def home():
    return {
        "message": "Student Performance API is Online.",
        "docs_url": "/docs"
    }

# 5. Create the Prediction End-point
@app.post("/predict")
def predict_performance(data: StudentDataInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized or available.")

    try:
        data_dict = {
            "Study_Balance": float(data.Study_Balance),
            "GPA_Difference": float(data.GPA_Difference),
            "Skill_Retention_Score": float(data.Skill_Retention_Score),
            "Anxiety_Level_During_Exams": float(data.Anxiety_Level_During_Exams),
            "Tool_Diversity": float(data.Tool_Diversity)
        }

        df_input = pd.DataFrame([data_dict])

        ordered_features = [
            "Study_Balance", 
            "GPA_Difference", 
            "Skill_Retention_Score", 
            "Anxiety_Level_During_Exams", 
            "Tool_Diversity"
        ]

        df_input = df_input[ordered_features]
        
        # 1. Get the actual class predicted by your machine learning model
        prediction_int = int(model.predict(df_input)[0])
        
        # 2. Extract the confidence score based on the TRUE prediction index
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_input)[0]
            high_performer_prob = float(probabilities[1])
            # 🔥 FIX: Lower the threshold from 0.50 down to 0.35 or 0.40
            # This allows a student with strong metrics to successfully cross the line!
            prediction_int = 1 if high_performer_prob >= 0.35 else 0
            confidence_score = high_performer_prob if prediction_int == 1 else probabilities[0]
        else:
            prediction_int = int(model.predict(df_input)[0])
            confidence_score = 1.0  
            
        # 3. Map output strings correctly (Assuming 1 = High, 0 = Standard)
        result_label = "High" if prediction_int == 1 else "Standard"
            
        return {
            "prediction_code": prediction_int,
            "prediction": result_label,
            "confidence": round(confidence_score, 4),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed pipeline execution: {str(e)}")