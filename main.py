from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np

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
        # Convert data into a 2D float32 array required by LightGBM
        input_features = np.array([[
            data.Study_Balance,
            data.GPA_Difference,
            data.Skill_Retention_Score,
            data.Anxiety_Level_During_Exams,
            data.Tool_Diversity
        ]], dtype=np.float32)
        
        # 1. Get the actual class predicted by your machine learning model
        prediction_int = int(model.predict(input_features)[0])
        
        # 2. Extract the confidence score based on the TRUE prediction index
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_features)[0]
            confidence_score = float(probabilities[prediction_int])
        else:
            confidence_score = 1.0  
            
        # 3. Map output strings correctly (Assuming 0 = High, 1 = Standard)
        result_label = "High" if prediction_int == 0 else "Standard"
            
        return {
            "prediction_code": prediction_int,
            "prediction": result_label,
            "confidence": round(confidence_score, 4),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed pipeline execution: {str(e)}")