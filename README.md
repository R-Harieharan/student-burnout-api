# Student Burnout Prediction & Dynamic API Optimization Pipeline

An end-to-end, production-ready machine learning pipeline and containerized FastAPI microservice built to forecast multi-class student burnout risk levels across 50,000 observation logs.

## Live Application Link

Click the direct endpoint link below to launch the production API control panel immediately:

  **[Launch Live Student Burnout API Dashboard](https://harie-06-student-burnout-api.hf.space/docs#/default/predict_burnout_predict_post)**

## Dataset Attribution
The raw behavioral observations utilized to train this architecture were sourced from the **[Kaggle Student Performance and Burnout Dataset](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students)**.

---

## Key Project Milestones

- **Data Ceiling Identification:** Recognized a strict data-noise threshold at ~48% validation accuracy within standard multi-class architectures. Built a custom threshold tuning pipeline using ordinal mapping to break past the ceiling and maximize modeling capacity.
- **Feature Space Optimization:** Trimmed feature bloat and multi-collinearity down to an optimal subset of 7 high-impact features selected directly from the random forest importance distributions, maximizing model interpretability and inference speed.
- **Overfitting Mitigation:** Successfully identified a training data memorization trap (reducing train accuracy from an unstable 100% down to a realistic 55%) by introducing sequential Gradient Boosting architectures (LightGBM).
- **Microservice Architecture Deployment:** Engineered a high-performance FastAPI backend endpoint that loads fully custom pipeline preprocessing states, handles out-of-vocabulary data exceptions gracefully, and executes live sub-second model inferences.

---

## Advanced Pipeline Architecture

Standard tabular data scripts often rely on naive, brute-force encoding strategies that destroy relational context. This project implements a split-preprocessing framework:
1. **Ordinal Mapping:** Preserves strict sequential progression rules for ordered variables (e.g., `Year_of_Study`, `Prompt_Engineering_Skill`).
2. **Custom Scikit-Learn Estimators:** Implements a robust `FeatureEngineer` component to calculate dynamic interaction deltas (e.g., GPA Shift over time) and an explicit `OutlierCapper` layer to stabilize scaling calculations.
3. **Continuous-to-Ordinal Bound Optimization:** Converted a loose, sparse classification task into a regression boundary lookup matrix using an `LGBMRegressor`. Custom grid-search optimizations established stable classification decision points:
   - **Low ➔ Medium Cutoff Bound:** `0.643`
   - **Medium ➔ High Cutoff Bound:** `1.271`

---

## Final Model Scorecard & Balanced Utility

Slicing the feature space by half removed secondary feature noise, lifting overall test accuracy to its peak performance of **50.74%**. The dynamic boundary shifting optimization unlocked exceptionally stable recall performance across all three distinct risk target groups, keeping extreme misclassifications to a minimum.


| Target Metric Group | Precision Score | Recall (Sensitivity) | F1-Score |
| :--- | :--- | :--- | :--- |
| **Low Risk** | 0.51 | 0.36 | 0.43 |
| **Medium Risk** | 0.47 | 0.67 | 0.55 |
| **High Risk (Critical)** | 0.65 | 0.43 | 0.51 |
| **Macro Average** | **0.54** | **0.48** | **0.50** |

---

## Running the Live Inference API Local Web Server

To stand up the FastAPI background listening engine locally:

```bash
# 1. Install production framework requirements
pip install fastapi uvicorn joblib lightgbm pandas scikit-learn==1.6.1 pydantic numpy

# 2. Boot up the service endpoint
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Sample Client Inference Payload Request
```python
import requests

url = "http://localhost:8000/predict"
sample_data = {
    "Major_Category": "Computer Science",
    "Year_of_Study": "Junior",
    "Primary_Use_Case": "Coding Assistance",
    "Prompt_Engineering_Skill": "Intermediate",
    "Institutional_Policy": "Allowed with Citation",
    "Pre_Semester_GPA": 3.4,
    "Post_Semester_GPA": 3.1,
    "Traditional_Study_Hours": 15,
    "Weekly_GenAI_Hours": 8,
    "Anxiety_Score": 7.5,
    "Weekly_Usage_Hours": 12,
    "Paid_Subscription": 1
}

response = requests.post(url, json=sample_data).json()
print(f"Status: {response['status']} | Risk Prediction: {response['predicted_burnout_risk']}")
```
