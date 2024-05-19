# # Importing Necessary Libraries
import os
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.models import Sequential
from prometheus_client import start_http_server, Summary
from prometheus_client import Counter, Histogram
import time

# Step 1
app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'App Request Count', ['app_name', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['app_name', 'endpoint'])

def load_model(path: str) -> Sequential:
    """
    Load the model from the supplied path on the disk and return the keras.src.engine.sequential.Sequential model.
    """
    if os.path.exists(path):
        from keras.models import load_model
        return load_model(path)
    else:
        return None
    
def predict_digit(model: Sequential, data_point: List[float]) -> str:
    """
    Take the image serialized as an array of 784 elements and returns the predicted digit as string.
    """
    data_point = np.array(data_point).reshape(1, 784)
    prediction = model.predict(data_point).argmax()
    return str(prediction)

@app.get('/')
def index():
    REQUEST_COUNT.labels(app_name='digit_prediction_app', endpoint='/').inc()
    return {'message': "Welcome to the Digit Prediction API!"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI model loading")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()

    app.model_path = args.model_path
    app.model = load_model(app.model_path)
    
    if app.model is None:
        print(f"Error: Model file not found at {app.model_path}")
    else:
        print(f"Model loaded successfully from {app.model_path}")

    start_http_server(8001)  # Start Prometheus metrics server
    # uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    REQUEST_COUNT.labels(app_name='digit_prediction_app', endpoint='/predict').inc()
    contents = await file.read()
    # Convert the image to grayscale and resize to 28x28
    img = Image.open(BytesIO(contents)).convert('L').resize((28, 28))
    # Convert image to array
    img_array = np.array(img).astype('float32') / 255.0
    # Flatten the array
    img_flat = img_array.flatten().tolist()
    # Load the model
    model = load_model("mnist_model.keras")
    # Make prediction
    prediction = predict_digit(model, img_flat)
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(app_name='digit_prediction_app', endpoint='/predict').observe(latency)
    return {"digit": prediction}

# Task 2

def format_image(image):
    # Convert the image to grayscale and resize to 28x28
    image_grey = image.convert('L').resize((28, 28))
    # Serialize the image as an array of 784 elements
    serial_array = list(image_grey.getdata())
    return serial_array

@app.post("/predictany")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    REQUEST_COUNT.labels(app_name='digit_prediction_app', endpoint='/predictany').inc()
    # Read the bytes from the uploaded image
    # await file.read()

    # Convert the bytes to a PIL Image
    image = Image.open(file.file)

    Serialized_image = format_image(image)
    
    # Load the model
    model = load_model("mnist_model.keras")
    
    # Get the predicted digit
    digit = predict_digit(model, Serialized_image)
    latency = time.time() - start_time
    REQUEST_LATENCY.labels(app_name='digit_prediction_app', endpoint='/predictany').observe(latency)
    # Return the predicted digit
    return {"digit": digit}

# Uncomment the following lines to run the application directly
# if __name__ == '__main__':
#     start_http_server(8001)  # Start Prometheus metrics server
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# import uvicorn
# from fastapi import FastAPI, File, UploadFile, Response, Request
# from keras.models import load_model
# from keras.models import Sequential
# import numpy as np
# from PIL import Image
# import argparse
# import ast
# import prometheus_client
# from prometheus_fastapi_instrumentator import Instrumentator
# from prometheus_client import Gauge,Summary,disable_created_metrics
# import time

# REQUEST_COUNT = prometheus_client.Counter('app_requests_total', 'Total HTTP requests processed', ['method', 'endpoint', 'client_ip'])
# # disable _created metric.
# disable_created_metrics()

# REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
# instrumentator = Instrumentator(
#     should_group_status_codes=False,
#     should_ignore_untemplated=True,
#     should_instrument_requests_inprogress=True,
#     inprogress_labels=True,
# )
# INPUT_LENGTH = Gauge('input_length', 'Length of the input text')
# TOTAL_TIME = Gauge('total_time_ms', 'Total time taken by the API in milliseconds')
# PROCESSING_TIME_PER_CHAR = Gauge('processing_time_per_char_us', 'Effective processing time per character in microseconds')
# # 1. Create a FastAPI module
# app = FastAPI()

# # 2. Take the path of the model as a command line argument.
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
# args = parser.parse_args()

# # 3. Create a function "def load_model(path:str) -> Sequential" which will load the model saved at the supplied path on the disk and return the keras.src.engine.sequential.Sequential model.
# def load_model_path(path: str) -> Sequential:
#     return load_model(path)

# # 4. Create a function "def predict_digit(model:Sequential, data_point:list) -> str" that will take the image serialized as an array of 784 elements and returns the predicted digit as string.
# def predict_digit(model: Sequential, data_point: list) -> str:
#     data_point = np.array(data_point).reshape(1, 784)
#     prediction = model.predict(data_point).argmax()
#     return str(prediction)

# # Load the model
# model = load_model_path(args.model_path)
# # TASK 1
# @app.post("/predictSerial")
# async def predict(Serialized_image):
#     start_time = time.time()
#     Serialized_image  = ast.literal_eval(Serialized_image)
#     # Get the predicted digit
#     digit = predict_digit(model, Serialized_image)
#     end_time = time.time()
#     total_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
#     # Calculate the effective processing time per character
#     # Measure the length of the input text
#     input_length = len(Serialized_image)
#     processing_time_per_char_us = (total_time_ms * 1000) / input_length  # Convert to microseconds
    
#     # Set the Gauge values
#     INPUT_LENGTH.set(input_length)
#     TOTAL_TIME.set(total_time_ms)
#     PROCESSING_TIME_PER_CHAR.set(processing_time_per_char_us)
#     # Return the predicted digit
#     return {"digit": digit}


# # TASK 2
# # Create a new function “def format_image” which will resize any uploaded images to a 28x28 greyscale image followed by creating a serialized array of 784 elements.
# def format_image(image):
#     # Convert the image to grayscale and resize to 28x28
#     image_grey = image.convert('L').resize((28, 28))
#     # Serialize the image as an array of 784 elements
#     serial_array = list(image_grey.getdata())
#     return serial_array


# # 5. Create an API endpoint "@app post('/predict')" that will read the bytes from the uploaded image to create a serialized array of 784 elements. The array shall be sent to 'predict_digit' function to get the digit. The API endpoint should return {"digit":digit"} back to the client.
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Read the bytes from the uploaded image
#     # await file.read()
#     start_time = time.time()
#     # Convert the bytes to a PIL Image
#     image = Image.open(file.file)

#     Serialized_image = format_image(image)
    
#     # Get the predicted digit
#     digit = predict_digit(model, Serialized_image)
#     end_time = time.time()
#     total_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
    
#     # Calculate the effective processing time per character
#     # Measure the length of the input text
#     input_length = len(Serialized_image)
#     processing_time_per_char_us = (total_time_ms * 1000) / input_length  # Convert to microseconds
    
#     # Set the Gauge values
#     INPUT_LENGTH.set(input_length)
#     TOTAL_TIME.set(total_time_ms)
#     PROCESSING_TIME_PER_CHAR.set(processing_time_per_char_us)
#     # Return the predicted digit
#     return {"digit": digit}

# # @app.get('/metrics')
# # def get_metrics():
# #     return Response(
# #         content=prometheus_client.generate_latest(),
# #         media_type="text/plain",
# #     )
# @app.middleware("http")
# async def count_requests(request: Request, call_next):
#     method = request.method
#     path = request.url.path
#     client_ip = request.client.host
    
#     response = await call_next(request)
    
#     REQUEST_COUNT.labels(method=method, endpoint=path, client_ip=client_ip).inc()
    
#     return response

# instrumentator.instrument(app).expose(app)
# # 6. Test the API via the Swagger UI (/docs) or Postman, where you will upload the digit as an image (28x28 size).
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)