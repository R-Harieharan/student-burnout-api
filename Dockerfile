FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Hugging Face Spaces run internally on port 7860
CMD ["uvicorn", "main", "--host", "0.0.0.0", "--port", "7860"]