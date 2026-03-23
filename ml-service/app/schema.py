from pydantic import BaseModel


class PredictionRequest(BaseModel):
    teamA: str
    teamB: str
    venue: str
    over: float
    ball: float


class PredictionResponse(BaseModel):
    score: float
    wickets: float
