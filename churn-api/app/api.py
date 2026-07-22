from fastapi import APIRouter

from app import model
from app.schemas import (
    FeatureImportancesResponse,
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model.load_model()
    return HealthResponse(status="ok", model_loaded=model.is_model_loaded())


@api_router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    result = model.predict_one(request.customer.model_dump())
    return PredictResponse(**result)


@api_router.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    customers = [c.model_dump() for c in request.customers]
    results = model.predict_many(customers)
    return PredictBatchResponse(predictions=[PredictResponse(**r) for r in results])


@api_router.get("/feature-importances", response_model=FeatureImportancesResponse)
def get_feature_importances() -> FeatureImportancesResponse:
    importances = model.feature_importances()
    return FeatureImportancesResponse(importances=importances)


@api_router.get("/model-info")
def get_model_info() -> dict:
    return model.model_info()