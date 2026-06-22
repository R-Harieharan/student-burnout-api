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
            "Study_Balance": float(data.Study_Balance / 40.0),
            "GPA_Difference": float(data.GPA_Difference),
            "Skill_Retention_Score": float(data.Skill_Retention_Score / 100.0),
            "Anxiety_Level_During_Exams": float(data.Anxiety_Level_During_Exams / 10.0),
            "Tool_Diversity": float(data.Tool_Diversity / 5.0)
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
        
        # 2. Extract the confidence score based on the TRUE prediction index
        if hasattr(model, "predict_proba"):
            # Get probabilities array for the single input row
            probabilities = model.predict_proba(df_input)           
            # Index 1 corresponds to "High Performer"
            high_performer_prob = float(probabilities[0][0])
            # Lower threshold down to 0.50 to offset dataset bias
            prediction_int = 1 if high_performer_prob >= 0.50 else 0            
            # Assign confidence based on the custom threshold result
            if prediction_int == 1:
                confidence_score = high_performer_prob
            else:
                confidence_score = float(probabilities[0][1])
        else:
            # Safe non-probability fallback
            raw_pred = model.predict(df_input)
            if hasattr(raw_pred, "__len__"):
                prediction_int = int(raw_pred[0])
            else:
                prediction_int = int(raw_pred)
            prediction_int = 1 if raw_val == 0 else 0
            confidence_score = 1.0  

        #result_label = f"DEBUG: Standard Prob is {probabilities[0][0]:.4f} and High Prob is {probabilities[0][1]:.4f}"
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