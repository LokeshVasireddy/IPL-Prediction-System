from fastapi import FastAPI
# from app.routes import router
# from core.model_loader import load_production_model

app = FastAPI()

# Include router first
# app.include_router(router)

# model_bundle = load_production_model()

@app.get("/")
def home():
    return {"message": "ML service running"}

@app.post("/predict")
def predict(data: dict):
    # dummy prediction
    # prediction = model_bundle.predict(data)
    return {
        "prediction": "Team A wins",
        "input": data
    }

@app.get("/health")
def health():
    return {"status": "ok"}