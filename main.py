import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_engineered = X.copy()
        if 'Post_Semester_GPA' in X_engineered.columns and 'Pre_Semester_GPA' in X_engineered.columns:
            X_engineered['GPA_Difference'] = X_engineered['Post_Semester_GPA'] - X_engineered['Pre_Semester_GPA']
            X_engineered = X_engineered.drop(columns=['Pre_Semester_GPA', 'Post_Semester_GPA'], errors='ignore')
        if 'Traditional_Study_Hours' in X_engineered.columns and 'Weekly_GenAI_Hours' in X_engineered.columns:
            X_engineered['Study_Balance'] = X_engineered['Traditional_Study_Hours'] - X_engineered['Weekly_GenAI_Hours']
            X_engineered = X_engineered.drop(columns=['Traditional_Study_Hours', 'Weekly_GenAI_Hours'], errors='ignore')
        if 'Student_ID' in X_engineered.columns:
            X_engineered = X_engineered.drop(columns=['Student_ID'], errors='ignore')
        return X_engineered
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        features = list(input_features)
        if 'Pre_Semester_GPA' in features and 'Post_Semester_GPA' in features:
            features.append('GPA_Difference')
            features.remove('Pre_Semester_GPA')
            features.remove('Post_Semester_GPA')
        if 'Traditional_Study_Hours' in features and 'Weekly_GenAI_Hours' in features:
            features.append('Study_Balance')
            features.remove('Traditional_Study_Hours')
            features.remove('Weekly_GenAI_Hours')
        if 'Student_ID' in features:
            features.remove('Student_ID')
        return np.array(features)

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, lower_percentile=0.01, upper_percentile=0.99):
        self.features = features
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.capping_values_ = {}
    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.select_dtypes(include=['number']).columns.tolist()
        for col in self.features:
            lower_bound = X[col].quantile(self.lower_percentile)
            upper_bound = X[col].quantile(self.upper_percentile)
            self.capping_values_[col] = {'lower': lower_bound, 'upper': upper_bound}
        return self
    def transform(self, X):
        X_transformed = X.copy()
        for col, bounds in self.capping_values_.items():
            X_transformed[col] = X_transformed[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array([])

sys.modules['__main__'].FeatureEngineer = FeatureEngineer
sys.modules['__main__'].OutlierCapper = OutlierCapper

app = FastAPI(
    title="STUDENT BURNOUT PREDICTION INTERACTIVE API",
    description="""
## Production-Grade ML Inference Architecture
This enterprise-ready microservice utilizes a threshold-optimized **LightGBM Regressor** pipeline.
""",
    version="1.4.2",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1, "docExpansion": "list"}
)

# Load model artifacts
try:
    model_artifacts = joblib.load('burnout_prediction_model_artifacts.joblib')
    full_preprocessing_pipeline = model_artifacts['preprocessing_pipeline']
    reg_model = model_artifacts['regression_model']
    target_map = model_artifacts['target_map']
    best_th1 = model_artifacts['best_th1']
    best_th2 = model_artifacts['best_th2']
    original_feature_names = model_artifacts['original_feature_names']
    top_7_features = model_artifacts['top_7_features']
    print("✅ Pipeline artifacts successfully loaded.")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    full_preprocessing_pipeline = None
    reg_model = None
    target_map = None
    best_th1 = None
    best_th2 = None
    original_feature_names = None
    top_7_features = None

# Create reverse mapping
reverse_target_map = {v: k for k, v in (target_map or {}).items()}

# Safety conversion for thresholds
if isinstance(best_th1, str):
    best_th1 = float(best_th1)
if isinstance(best_th2, str):
    best_th2 = float(best_th2)

@app.post("/predict", summary="Execute Inference Pipeline", responses={422: {"description": "Validation Error Disabled"}})
def predict_burnout(student_data: dict = Body(...)):
    if full_preprocessing_pipeline is None or reg_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # Create DataFrame (preserve original dtypes)
        df_input = pd.DataFrame([student_data])

        # Add missing columns
        for col in original_feature_names:
            if col not in df_input.columns:
                df_input[col] = np.nan

        # Reorder columns exactly as training
        df_input = df_input[original_feature_names]

        # === RUN PIPELINE ===
        processed_data = full_preprocessing_pipeline.transform(df_input)

        # Extract top 7 features safely
        if isinstance(processed_data, np.ndarray):
            try:
                feature_names_out = full_preprocessing_pipeline.get_feature_names_out()
                clean_cols = [c.split('__')[-1] if '__' in c else c for c in feature_names_out]
                processed_df = pd.DataFrame(processed_data, columns=clean_cols)
                processed_data = processed_df[top_7_features].values
            except Exception:
                processed_data = processed_data[:, :len(top_7_features)]
        elif isinstance(processed_data, pd.DataFrame):
            processed_data = processed_df[top_7_features].values
        else:
            processed_data = np.asarray(processed_data)[:, :len(top_7_features)]

        # Predict regression score
        raw_score = float(reg_model.predict(processed_data)[0])

        # Robust thresholding
        th1 = float(best_th1) if best_th1 is not None else 0.643
        th2 = float(best_th2) if best_th2 is not None else 1.271

        if raw_score < th1:
            predicted_class_id = 0
        elif raw_score < th2:
            predicted_class_id = 1
        else:
            predicted_class_id = 2

        risk_level = reverse_target_map.get(predicted_class_id, "Unknown")

        return {
            "status": "success",
            "raw_regression_score": round(raw_score, 4),
            "predicted_burnout_risk": risk_level,
            "applied_threshold_bounds": {
                "low_cutoff": round(th1, 3),
                "high_cutoff": round(th2, 3)
            }
        }

    except Exception as e:
        import traceback
        error_detail = f"Pipeline inference failure: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=str(e))