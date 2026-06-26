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
    model = joblib.load("student_performance_lgbm_model.pkl")
    # Initialize the SHAP explainer for LightGBM tree-based models
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
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
            standard_prob = float(probabilities[0][1])
            high_prob = float(probabilities[0][0])
            if high_prob > 0.51:
                prediction_int = 1
                result_label = "High"
                confidence_score = high_prob
            else:
                prediction_int = 0
                result_label = "Standard"
                confidence_score = standard_prob
            if (
                float(data.Skill_Retention_Score) < 50.0
                and float(data.Anxiety_Level_During_Exams) >= 7.0
            ):
                prediction_int = 0
                result_label = "Standard"
                confidence_score = standard_prob    
        else:
            raw_pred = model.predict(df_input)
            flat_pred = raw_pred.flatten() if hasattr(raw_pred, "flatten") else raw_pred
            raw_val = int(flat_pred[0]) if hasattr(flat_pred, "__len__") else int(flat_pred)
            if raw_val == 1:
                prediction_int = 1
                result_label = "High"
            else:
                prediction_int = 0
                result_label = "Standard"
            confidence_score = 1.0

        # --- SHAP Explanation Generation & Scalar Optimization ---
        plot_base64 = ""
        try:
            # 1. Use the modern call syntax to extract structured explanations
            shap_values = explainer(df_input)
            
            # 2. Slice the matrix safely based on the dimension shape
            if len(shap_values.shape) == 3:
                # Multi-class layout: [row_index, feature_index, class_index]
                shap_values_display = shap_values[0, :, prediction_int]
            elif len(shap_values.shape) == 2:
                # Binary single-output layout: [row_index, feature_index]
                shap_values_display = shap_values[0, :]
            else:
                shap_values_display = shap_values

            # 3. CRITICAL FIXED: Force base_values to be a scalar float to satisfy waterfall rules
            if hasattr(shap_values_display, "base_values"):
                bv = shap_values_display.base_values
                # Flatten and extract the scalar value if it's locked inside an array wrapper
                if hasattr(bv, "__len__") or isinstance(bv, np.ndarray):
                    shap_values_display.base_values = float(np.ravel(bv)[0])
                else:
                    shap_values_display.base_values = float(bv)

            # 4. Generate and save the waterfall plot to memory buffer
            plt.figure(figsize=(8, 4))
            shap.plots.waterfall(shap_values_display, show=False)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()
            
        except Exception as plot_err:
            # Captures any inner formatting notices safely to your server logs
            print(f"SHAP Vector Plotting Fix Notice: {str(plot_err)}")
            plot_base64 = ""


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
        