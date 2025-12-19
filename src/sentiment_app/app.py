from fastapi import FastAPI

from .settings import Settings
from .inference import Inference

from pydantic import BaseModel
from mangum import Mangum


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str


app = FastAPI()
settings = Settings()
inference = Inference(settings)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction=inference.predict(request.text))


handler = Mangum(app)
