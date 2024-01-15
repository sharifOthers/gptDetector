from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from mangum import Mangum
import tensorflow as tf
from tensorflow import keras


model = tf.keras.models.load_model('trained_model')

app = FastAPI()
handler = Mangum(app)


@app.get("/")
async def root():
    return {"message": "Welcome to GptDetector"}


@app.get("/detect_gpt")
async def detect_gpt(text: str):
    prediction = model.predict([text])
    if prediction[0][0] >= 0.5:
        return {"message":"Generate Text"}
    else:
        return {"message":"Not Generated Text"}
