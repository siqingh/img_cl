from fastapi import FastAPI
from fastapi import File
from pydantic import BaseModel
from app.model.inference import predict_furniture

app = FastAPI()

class PredictedLabel(BaseModel):
    furniture_type: str

@app.get("/")
async def home():
    return {"status": "OK"}

@app.post("/api/classify", response_model=PredictedLabel)
async def classify(payload: bytes = File(...)):
    furniture = predict_furniture(payload)
    return {"furniture_type": furniture}