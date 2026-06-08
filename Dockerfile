FROM python:3.9-slim

# Set the working directory
WORKDIR /code

# Install system dependencies required for LightGBM and other ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Start the FastAPI application
# Ensure 'main:app' matches your actual file structure
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]