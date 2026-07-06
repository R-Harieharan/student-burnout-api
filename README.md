# 🚀 Student Burnout Prediction API

A production-ready FastAPI backend serving machine learning predictions for the **Student Performance & Burnout Analytics Dashboard**.

This REST API exposes a LightGBM classification model capable of predicting student burnout risk from engineered academic features while providing prediction confidence and probability estimates.

---

## Overview

The API acts as the inference layer for the Student Performance & Burnout Analytics Dashboard.

It is responsible for:

- Loading the production-trained LightGBM model
- Validating incoming requests
- Running real-time predictions
- Returning prediction confidence
- Providing probability estimates
- Serving a documented REST API

The backend is designed with modularity, reproducibility, and deployment simplicity in mind.

---

## System Architecture

```text
                Client Application
                       │
                       ▼
              Streamlit Dashboard
                       │
                 HTTP POST Request
                       │
                       ▼
                FastAPI REST API
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Input Validation   Feature Parser   Error Handling
        │
        ▼
   LightGBM Model
        │
        ▼
Prediction + Probabilities
        │
        ▼
 JSON Response
```

---

# Features

## REST API

- FastAPI framework
- OpenAPI documentation
- Interactive Swagger UI
- Typed request validation

---

## Machine Learning

- Production LightGBM classifier
- Joblib model serialization
- Probability prediction
- Confidence estimation

---

## Validation

- Pydantic request models
- Automatic input validation
- Type safety
- Structured error responses

---

## Deployment

- Hugging Face Spaces
- Docker compatible
- Production inference endpoint

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| API Framework | FastAPI |
| Validation | Pydantic |
| ML Model | LightGBM |
| Serialization | Joblib |
| Data Processing | Pandas |
| Deployment | Hugging Face Spaces |

---

# API Workflow

```text
Client Request
      │
      ▼
Pydantic Validation
      │
      ▼
Feature Extraction
      │
      ▼
LightGBM Prediction
      │
      ▼
Probability Estimation
      │
      ▼
JSON Response
```

---

# Repository Structure

```text
student-burnout-api/

├── app.py
├── models/
│   ├── student_performance_lgbm_model.pkl
│   └── production_features_list.pkl
│
├── requirements.txt
├── Dockerfile
├── README.md
├── burnout_risk_prediction.py
└── ...
```

---

# API Endpoint

## Predict Burnout Risk

```
POST /predict
```

---

## Sample Request

```json
{
  "Study_Balance": 0.75,
  "GPA_Difference": 0.32,
  "Skill_Retention_Score": 0.81,
  "Anxiety_Level_During_Exams": 0.45,
  "Tool_Diversity": 0.60
}
```

---

## Sample Response

```json
{
  "prediction_code": 1,
  "prediction": "High",
  "confidence": 0.9137,
  "high_probability": 0.9137,
  "not_high_probability": 0.0863,
  "status": "Success"
}
```

---

# API Documentation

Once the server is running, interactive documentation is available through Swagger.

```
/docs
```

The documentation includes

- request schema
- response schema
- example payloads
- endpoint testing

---

# Local Installation

Clone the repository

```bash
git clone https://github.com/R-Harieharan/student-burnout-api.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the API

```bash
uvicorn app:app --reload
```

The API will be available at

```
http://127.0.0.1:8000
```

---

# Production Deployment

The backend is deployed using **Hugging Face Spaces** and serves as the prediction engine for the Student Performance & Burnout Analytics Dashboard.

---

# Frontend Integration

This API powers the frontend dashboard available here:

**Frontend Repository**

https://github.com/R-Harieharan/student-performance-burnout-dashboard

The frontend communicates with this backend through REST API requests to perform real-time student burnout prediction.

---

# Error Handling

The API returns meaningful HTTP status codes.

| Status Code | Description |
|-------------|-------------|
| 200 | Prediction successful |
| 400 | Invalid request |
| 422 | Validation error |
| 500 | Prediction pipeline failure |
| 503 | Model unavailable |

---

# Model Information

Production Model

- Algorithm: LightGBM Classifier
- Task: Binary Classification
- Output: Burnout Risk Prediction
- Probability Estimates: Enabled
- Confidence Score: Enabled

---

# Future Improvements

Potential future enhancements include:

- Batch prediction endpoint
- Authentication
- API versioning
- Request logging
- Monitoring and metrics
- Model version management
- Rate limiting

---

# License

MIT License

---

# Author

**Harie**

Computer Science Student

GitHub:
https://github.com/R-Harieharan

Frontend Repository:
https://github.com/R-Harieharan/student-performance-burnout-dashboard

Hugging Face:
https://huggingface.co/Harie-06
