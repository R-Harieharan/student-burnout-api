import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

sys.modules['__main__'].FeatureEngineer = FeatureEngineer
sys.modules['__main__'].OutlierCapper = OutlierCapper

app = FastAPI(
    title="Student Burnout Prediction API",
    description="Production inference endpoint utilizing threshold-optimized 7-feature regression.",
    version="1.1.0"
)

try:
    model_artifacts = joblib.load('burnout_prediction_model_artifacts.joblib')
    full_preprocessing_pipeline = model_artifacts['preprocessing_pipeline']
    reg_model = model_artifacts['regression_model']
    target_map = model_artifacts['target_map']
    best_th1 = model_artifacts['best_th1']
    best_th2 = model_artifacts['best_th2']
    original_feature_names = model_artifacts['original_feature_names']
    top_7_features = model_artifacts['top_7_features']

    print("Pipeline artifacts successfully loaded into environment.")
except Exception as e:
    print(f"Execution halted during serialization unpacking: {str(e)}")
    exit()

reverse_target_map = {v: k for k, v in target_map.items()}

class StudentInput(BaseModel):
    model_config = {"extra": "allow"}

@app.post("/predict", summary="Execute Inference Pipeline")
def predict_burnout(student_data: dict):
    try:
        df_input = pd.DataFrame([student_data])
        
        for col in original_feature_names:
            if col not in df_input.columns:
                df_input[col] = np.nan
        
        df_input = df_input[original_feature_names]
        processed_data = full_preprocessing_pipeline.transform(df_input)
        
        # Convert numpy array back to DataFrame using pipeline names 
        # to ensure safe string-based feature matching
        if isinstance(processed_data, np.ndarray):
            feature_names_out = full_preprocessing_pipeline.get_feature_names_out()
            # Standardize names by removing standard scikit-learn encoder prefixes if present
            clean_cols = [c.split('__')[-1] for c in feature_names_out]
            processed_df = pd.DataFrame(processed_data, columns=clean_cols)
            processed_data = processed_df[top_7_features].values
        elif isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data[top_7_features].values
            
        raw_score = float(reg_model.predict(processed_data))
        
        if raw_score < best_th1:
            predicted_class_id = 0
        elif raw_score < best_th2:
            predicted_class_id = 1
        else:
            predicted_class_id = 2
            
        risk_level = reverse_target_map.get(predicted_class_id, "Unknown")
        
        return {
            "status": "success",
            "raw_regression_score": round(raw_score, 4),
            "predicted_burnout_risk": risk_level,
            "applied_threshold_bounds": {
                "low_cutoff": round(best_th1, 3), 
                "high_cutoff": round(best_th2, 3)
            }
        }
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Pipeline inference failure: {str(e)}")
