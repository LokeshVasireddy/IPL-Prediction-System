from fastapi import APIRouter
from app.schema import PredictionRequest, PredictionResponse
from app.inference import run_prediction

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    return run_prediction(data.dict())