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

        # --- SHAP Explanation Generation & Isolated Safe Plotting Block ---
        plot_base64 = ""
        try:
            # 1. Extract raw metrics
            raw_shap = explainer.shap_values(df_input)
            base_vals = explainer.expected_value
            
            # 2. Force feature impacts down to a pure 1D array
            if isinstance(raw_shap, list):
                row_shap = np.array(raw_shap[prediction_int]).flatten()
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 3:
                    row_shap = raw_shap[0, :, prediction_int].flatten()
                else:
                    row_shap = raw_shap.flatten()
            else:
                row_shap = np.array(raw_shap).flatten()
                
            # 3. Force base values down to a pure scalar float
            if isinstance(base_vals, (list, np.ndarray)):
                if len(base_vals) > prediction_int:
                    base_value = float(base_vals[prediction_int])
                else:
                    base_value = float(base_vals[0])
            else:
                base_value = float(base_vals)
            
            # 4. FIXED: Flatten input data to 1D to prevent dimension crashes and text truncations
            flat_data = np.array(df_input.values, dtype=float).flatten()
            
            # 5. Build the pristine 1D Explanation structure
            shap_values_display = shap.Explanation(
                values=np.array(row_shap, dtype=float),
                base_values=float(base_value),
                data=flat_data, # <-- Pure 1D array resolves row-parsing blocks
                feature_names=list(ordered_features)
            )
                
            # 6. FIXED: Lock a static canvas size and drop tight_layout to stop the shaking bug completely
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=100)
            shap.plots.waterfall(shap_values_display, show=False)
            
            # Save the rendering directly into memory 
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.2)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close(fig) # Completely free up thread memory
            
        except Exception as plot_err:
            print(f"SHAP Layout Framework Notice: {str(plot_err)}")
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
        