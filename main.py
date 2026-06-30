from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Global variables placeholder
model = None

# 1. Define modern Lifespan handler for model and explainer loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load("student_performance_lgbm_model.pkl")
    except Exception as e:
        raise RuntimeError(f"Could not load the model. Error: {str(e)}")
    yield
    # Clean up operations can go here if needed on shutdown

# 2. Initialize the FastAPI Application with lifespan
app = FastAPI(
    title="Student Performance Prediction API",
    description="Production-ready API for student burnout risk prediction."
    version="1.1",
    root_path="/proxy",
    lifespan=lifespan,
)

# 3. Define the Input Data Schema using clean Pydantic V2 syntax
class StudentDataInput(BaseModel):
    Study_Balance: float = Field(
        ..., description="Calculated student study balance metric", examples=[0.75]
    )
    GPA_Difference: float = Field(
        ..., description="Difference margin in student GPA scores", examples=[0.32]
    )
    Skill_Retention_Score: float = Field(
        ..., description="Testing retention score metric", examples=[0.81]
    )
    Anxiety_Level_During_Exams: float = Field(
        ..., description="Normalized metric for exam anxiety", examples=[0.45]
    )
    Tool_Diversity: float = Field(
        ...,
        description="Score metric representing tech tool diversity utilized",
        examples=[0.60],
    )

# 4. Create the Home Route
@app.get("/")
def home():
    return {"message": "Student Performance API is Online.", "docs_url": "/docs"}

# 5. Create the Prediction End-point
@app.post("/predict")
def predict_performance(data: StudentDataInput):
    try:
        if model is None:
            raise HTTPException(
                status_code=503, detail="Model is not available."
            )
        
        data_dict = {
            "Study_Balance": float(data.Study_Balance),
            "GPA_Difference": float(data.GPA_Difference),
            "Skill_Retention_Score": float(data.Skill_Retention_Score),
            "Anxiety_Level_During_Exams": float(data.Anxiety_Level_During_Exams),
            "Tool_Diversity": float(data.Tool_Diversity),
        }
        df_input = pd.DataFrame([data_dict])
    
        # --- Prediction & Confidence Code ---
        prediction_int = 0 # Default to Not High
        result_label = "Not High"
        confidence_score = 0.0
        high_prob = 0.0
        not_high_prob = 0.0
    
        if hasattr(model, "predict_proba"):
    
            prediction_int = int(model.predict(df_input)[0])
            probabilities = model.predict_proba(df_input)[0]
            # Assuming 0 is 'Not High' and 1 is 'High'
            not_high_prob = float(probabilities[0])
            high_prob = float(probabilities[1])
    
            confidence_score = (
                high_prob if prediction_int == 1 else not_high_prob
            )
            
            result_label = "High" if prediction_int == 1 else "Not High"

            return {
                "prediction_code": prediction_int,
                "prediction": result_label,
                "confidence": round(confidence_score, 4),
                "high_probability": round(high_prob, 4),
                "not_high_probability": round(not_high_prob, 4),
                "status": "Success"
            }

    except HTTPException:
        raise
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed pipeline execution: {str(e)}",
        )