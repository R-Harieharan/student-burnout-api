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
        # Using feature_perturbation="tree_path_dependent" for consistency with TreeExplainer
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    except Exception as e:
        raise RuntimeError(f"Could not load the model or explainer file. Error: {str(e)}")
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
            
            """if high_prob > not_high_prob: # Decide based on higher probability
                prediction_int = 1
                result_label = "High"
                confidence_score = high_prob
            else:
                prediction_int = 0
                result_label = "Not High"
                confidence_score = not_high_prob

        else:
            # Fallback if probability isn't supported
            raw_pred = model.predict(df_input)
            flat_pred = raw_pred.flatten() if hasattr(raw_pred, "flatten") else raw_pred
            prediction_int = int(flat_pred[0]) if hasattr(flat_pred, "__len__") else int(flat_pred)

            if prediction_int == 1:
                result_label = "High"
            else:
                result_label = "Not High"
            confidence_score = 1.0 # Cannot determine confidence without proba"""

        # --- SHAP Explanation Generation & Custom Type-Aware Native Plotting ---
        plot_base64 = ""
        row_shap = [] # Initialize row_shap for consistent return
        try:
            # 1. Extract raw SHAP feature impact array weights cleanly
            raw_shap = explainer.shap_values(df_input)
            
            # 2. DYNAMICALLY DETECT AND EXTRACT 1D FEATURE IMPACTS
            # Fixes the IndexError by checking if SHAP returned a multi-class list or a flat 1D binary array
            if isinstance(raw_shap, list):
                # For multi-class or binary classification, shap_values returns a list.
                # We want the SHAP values for the predicted class.
                row_shap = np.array(raw_shap[prediction_int]).flatten()
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 3: # e.g., (1, num_features, num_classes) for some explainers
                    row_shap = raw_shap[0, :, prediction_int].flatten()
                elif len(raw_shap.shape) == 2 and raw_shap.shape[0] == 1: # e.g., (1, num_features)
                    row_shap = raw_shap[0].flatten()
                else:
                    # If it's already a flat 1D binary array of shape (num_features,), use it directly!
                    row_shap = raw_shap.flatten()
            else:
                # Fallback, attempt to convert to numpy array and flatten
                row_shap = np.array(raw_shap).flatten()
               
            # 3. Read the clean raw numerical student inputs array
            flat_data = np.array(df_input.values, dtype=float).flatten()
            
            # 4. Generate clean display labels combining feature name + user inputs
            display_labels = []
            for feat, val in zip(ordered_features, flat_data):
                display_labels.append(f"{feat} = {val:.2f}") # Format to 2 decimal places
                
            # 5. Use Matplotlib's Object-Oriented Canvas to completely isolate drawing memory threads
            fig = matplotlib.figure.Figure(figsize=(8.0, 4.0), dpi=100)
            fig.subplots_adjust(left=0.35, right=0.92, top=0.85, bottom=0.18)
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
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Release allocations immediately to maintain fast response loops
            buffer.close()
            plt.close(fig) # Explicitly close the figure to free memory
            del fig, ax
            
        except Exception as plot_err:
            # Stream error strings directly to your space logs console for easy debugging
            print(f"CRITICAL CUSTOM PLOT LOG NOTICE: {str(plot_err)}")
            plot_base64 = ""
            row_shap = [0.0] * len(ordered_features) # Ensure row_shap is a list of floats of correct length

        return {
            "prediction_code": prediction_int,
            "prediction": result_label,
            "confidence": round(confidence_score, 4),
            "high_probability": round(high_prob, 4),
            "not_high_probability": round(not_high_prob, 4),
            "shap_plot_base64": plot_base64,
            "shap_values": row_shap.tolist(), # Convert numpy array to list for JSON serialization
            "feature_names": ordered_features,
            "status": "Success",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed pipeline execution: {str(e)}",
        )