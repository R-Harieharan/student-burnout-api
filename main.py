%%writefile app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Student Performance Prediction API",
    description="A production-ready API to predict high-performing student outcomes using an optimized LightGBM model.",
    version="1.0"
)

# 2. Load the trained model safely on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("student_performance_lgbm_model.pkl")
    except Exception as e:
        print(f"CRITICAL: Model load failed: {str(e)}")


# 3. Define the Input Data Schema using Pydantic
# This ensures incoming API requests have the exact data types required
class StudentDataInput(BaseModel):
    Study_Balance: float = Field(..., description="Calculated student study balance metric", example=0.75)
    GPA_Difference: float = Field(..., description="Difference margin in student GPA scores", example=0.32)
    Skill_Retention_Score: float = Field(..., description="Testing retention score metric", example=0.81)
    Anxiety_Level_During_Exams: float = Field(..., description="Normalized metric for exam anxiety", example=0.45)
    Tool_Diversity: float = Field(..., description="Score metric representing tech tool diversity utilized", example=0.60)

# 4. Create the Home Route
@app.get("/")
def home():
    return {
        "message": "Student Performance API is Online.",
        "docs_url": "/docs"  # Direct link to FastAPI's interactive UI
    }

# 5. Create the Prediction End-point
@app.post("/predict")
def predict_performance(data: StudentDataInput):
    try:
        # Convert incoming JSON data into the exact 2D array order expected by the model
        input_features = np.array([[
            data.Study_Balance,
            data.GPA_Difference,
            data.Skill_Retention_Score,
            data.Anxiety_Level_During_Exams,
            data.Tool_Diversity
        ]])

        # Generate prediction (0 = Not High, 1 = High)
        prediction_int = int(model.predict(input_features)[0])

        # Get raw probability percentage scores for deeper production context
        probabilities = model.predict_proba(input_features)[0]
        confidence_score = float(probabilities[prediction_int])

        # Map output to user-friendly strings
        result_label = "High" if prediction_int == 1 else "Not High"

        return {
            "prediction_code": prediction_int,
            "prediction": result_label,
            "confidence": round(confidence_score, 4),
            "status": "Success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed pipeline execution: {str(e)}")
