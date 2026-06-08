---
title: Student Burnout Prediction
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Student Burnout Prediction & Dynamic API Optimization Pipeline

An end-to-end, production-ready machine learning pipeline and containerized FastAPI microservice built to forecast multi-class student burnout risk levels.

## 🚀 Access the API
**[Launch Live Student Burnout API Dashboard](https://huggingface.co/spaces/Harie-06/student-burnout-api)**

*Use the `/docs` endpoint in the URL above to interact directly with the prediction model.*

---

## 📊 Project Overview
This project transforms raw behavioral observation data into actionable burnout risk insights. By implementing custom Scikit-Learn estimators and gradient boosting architectures, the pipeline effectively navigates data-noise thresholds to provide stable multi-class classification.

### Key Milestones
- **Feature Optimization:** Reduced feature bloat to 7 high-impact variables, improving inference speed and model interpretability.
- **Overfitting Mitigation:** Transitioned from unstable architectures to sequential Gradient Boosting (LightGBM) to ensure realistic performance.
- **Custom Pipeline:** Engineered robust `FeatureEngineer` and `OutlierCapper` layers to handle real-world data variability.

---

## 📈 Model Performance
| Target Metric Group | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Low Risk** | 0.51 | 0.36 | 0.43 |
| **Medium Risk** | 0.47 | 0.67 | 0.55 |
| **High Risk (Critical)** | 0.65 | 0.43 | 0.51 |
| **Macro Average** | **0.54** | **0.48** | **0.50** |

---

## 🛠 Deployment & Local Usage

### Running Locally
To launch the FastAPI service on your local machine:

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn joblib lightgbm pandas scikit-learn==1.6.1 pydantic numpy