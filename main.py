import base64

import io

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import shap
# Force matplotlib to run without a GUI (Required for production servers)
matplotlib.use("Agg")
# Global variables placeholder
model = None
explainer = None
ordered_features = [
    "Study_Balance",
    "GPA_Difference",
    "Skill_Retention_Score",
    "Anxiety_Level_During_Exams",
    "Tool_Diversity",
]
# 1. Define modern Lifespan handler for model and explainer loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer
    try:
        model = joblib.load("student_performance_lgbm_model.pkl")
        # Initialize the SHAP explainer for LightGBM tree-based models
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        raise RuntimeError(
            f"Could not load the model or SHAP explainer. Error: {str(e)}"
        )
        yield
        # Clean up operations can go here if needed on shutdown
# 2. Initialize the FastAPI Application with lifespan
app = FastAPI(
    title="Student Performance Prediction API",
    description="A production-ready API with SHAP explanations to predict student outcomes.",
    version="1.1",
    root_path="/proxy",
    lifespan=lifespan,
)
# 3. Define the Input Data Schema using clean Pydantic V2 syntax
class StudentDataInput(BaseModel):
    Study_Balance: float = Field(
        ..., description="Calculated student study balance metric", examples=[35.0]
    )
    GPA_Difference: float = Field(
        ..., description="Difference margin in student GPA scores", examples=[0.8]
    )
    Skill_Retention_Score: float = Field(
        ..., description="Testing retention score metric", examples=[90.0]
    )
    Anxiety_Level_During_Exams: float = Field(
        ..., description="Normalized metric for exam anxiety", examples=[2.0]
    )
    Tool_Diversity: float = Field(
        ...,
        description="Score metric representing tech tool diversity utilized",
        examples=[4.0],
    )
# 4. Create the Home Route
@app.get("/")
def home():
    return {"message": "Student Performance API is Online.", "docs_url": "/docs"}
# 5. Create the Prediction End-point
@app.post("/predict")
def predict_performance(data: StudentDataInput):
    if model is None or explainer is None:
        raise HTTPException(
            status_code=503, detail="Model or SHAP explainer is not available."
        )
    try:
        data_dict = {
            "Study_Balance": float(data.Study_Balance),
            "GPA_Difference": float(data.GPA_Difference),
            "Skill_Retention_Score": float(data.Skill_Retention_Score),
            "Anxiety_Level_During_Exams": float(data.Anxiety_Level_During_Exams),
            "Tool_Diversity": float(data.Tool_Diversity),
        }
        df_input = pd.DataFrame([data_dict])[ordered_features]
        # --- Prediction & Confidence Code ---
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_input)
            flat_probs = probabilities.flatten()
            standard_prob = float(flat_probs[1])
            high_prob = float(flat_probs[0])
            prediction_int = 1 if high_prob > 0.51 else 0
            if (
                float(data.Skill_Retention_Score) < 50.0
                and float(data.Anxiety_Level_During_Exams) >= 7.0
            ):
                prediction_int = 0
                confidence_score = high_prob if prediction_int == 1 else standard_prob
            else:
                raw_pred = model.predict(df_input)
                flat_pred = (
                    raw_pred.flatten() if hasattr(raw_pred, "flatten") else raw_pred
                )
                raw_val = (
                    int(flat_pred[0]) if hasattr(flat_pred, "__len__") else int(flat_pred)
                )
                prediction_int = 1 if raw_val == 1 else 0
                confidence_score = 1.0
            result_label = "High" if prediction_int == 1 else "Standard"
            # --- SHAP Explanation Generation ---
            # Generate SHAP values for the single input row
            shap_values = explainer(df_input)
            # Handle probability output slicing if LightGBM returns multi-class shapes
            if len(shap_values.shape) == 3:
                # Slice for class index prediction_int to show factors driving this specific outcome
                shap_values_display = shap_values[0, :, prediction_int]
            else:
                shap_values_display = shap_values[0]
            # Generate a waterfall summary plot for the single prediction instance
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_values_display, show=False)
            plt.tight_layout()
            # Save plot to an in-memory buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()  # Clean up memory resources
            
            return {
                "prediction_code": prediction_int,
                "prediction": result_label,
                "confidence": round(confidence_score, 4),
                "shap_plot_base64": plot_base64,
                "status": "Success",
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed pipeline execution: {str(e)}",
        )
        