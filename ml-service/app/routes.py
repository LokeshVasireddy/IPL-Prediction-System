from app.inference import run_prediction
from app.schema import PredictionRequest, PredictionResponse
from fastapi import APIRouter

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    return run_prediction(data.dict())
