FROM python:3.10-slim

WORKDIR /code

# 1. Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Hugging Face Spaces run internally on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]