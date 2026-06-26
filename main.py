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

        # --- SHAP Explanation Generation & Custom Type-Aware Native Plotting ---
        plot_base64 = ""
        try:
            # 1. Extract raw SHAP feature impact array weights cleanly
            raw_shap = explainer.shap_values(df_input)
            
            # 2. DYNAMICALLY DETECT AND EXTRACT 1D FEATURE IMPACTS
            # Fixes the IndexError by checking if SHAP returned a multi-class list or a flat 1D binary array
            if isinstance(raw_shap, list):
                row_shap = np.array(raw_shap[prediction_int]).flatten()
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 3:
                    row_shap = raw_shap[0, :, prediction_int].flatten()
                elif len(raw_shap.shape) == 2:
                    row_shap = raw_shap[0].flatten()
                else:
                    # If it's already a flat 1D binary array of shape (5,), use it directly!
                    row_shap = raw_shap.flatten()
            else:
                row_shap = np.array(raw_shap).flatten()
            
            # 3. Read the clean raw numerical student inputs array
            flat_data = np.array(df_input.values, dtype=float).flatten()
            
            # 4. Generate clean display labels combining feature name + user inputs
            display_labels = []
            for feat, val in zip(ordered_features, flat_data):
                display_labels.append(f"{feat} = {val:.1f}")
                
            # 5. Use Matplotlib's Object-Oriented Canvas to completely isolate drawing memory threads
            fig = matplotlib.figure.Figure(figsize=(7.5, 3.8), dpi=100)
            ax = fig.add_subplot(111)
            
            # 6. Apply clear dual-color indicator bars (Royal Blue = Positive impact, Crimson = Negative impact)
            bar_colors = ['#1E3A8A' if val >= 0 else '#B91C1C' for val in row_shap]
            
            # Draw pristine horizontal impact indicators safely on the isolated canvas 
            y_positions = np.arange(len(ordered_features))
            ax.barh(y_positions, row_shap, color=bar_colors, edgecolor='none', height=0.6)
            
            # Configure visual frame metrics cleanly
            ax.set_yticks(y_positions)
            ax.set_yticklabels(display_labels, fontsize=10, fontweight='bold', color='#1F2937')
            ax.axvline(x=0, color='#6B7280', linestyle='--', linewidth=1)
            ax.set_xlabel("Feature Impact Weight (SHAP)", fontsize=10, fontweight='bold')
            ax.set_title(f"Performance Driver Breakdown ({result_label} Tier)", fontsize=11, fontweight='bold', pad=12)
            
            # Clean up borders to optimize whitespace layout
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#D1D5DB')
            ax.spines['bottom'].set_color('#D1D5DB')
            
            # 7. Save plot payload directly to a secure RAM byte matrix string stream
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Release allocations immediately to maintain fast response loops
            buffer.close()
            fig.clf()
            del fig, ax
            
        except Exception as plot_err:
            # Stream error strings directly to your space logs console for easy debugging
            print(f"CRITICAL CUSTOM PLOT LOG NOTICE: {str(plot_err)}")
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
        