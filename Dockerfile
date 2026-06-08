FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
