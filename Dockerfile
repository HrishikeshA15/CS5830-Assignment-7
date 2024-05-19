FROM python:3.10.2-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY Prometheus.py .

COPY mnist_model.keras .

EXPOSE 8000

EXPOSE 9090

ENTRYPOINT ["python","Prometheus.py","--model_path","mnist_model.keras"]