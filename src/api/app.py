from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

REGISTERED_MODEL_NAME = "claimsai_xgb_anomaly"

app = FastAPI(
    title="ClaimsAI Inference API",
    description="API for scoring healthcare claims anomalies",
    version="0.1.0",
)

_model: Optional[mlflow.pyfunc.PyFuncModel] = None
_model_uri: Optional[str] = None


# ------------------------------------------------------------------
# SCHEMAS
# ------------------------------------------------------------------


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(
        ..., description="List of raw feature rows (dicts)"
    )


class PredictResponse(BaseModel):
    model_uri: str
    predictions: List[int]
    scores: List[float]


# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------


def load_latest_model() -> None:
    """
    Load the latest registered model version from MLflow Model Registry.
    """
    global _model, _model_uri

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"No registered model found with name '{REGISTERED_MODEL_NAME}'"
        )

    latest = max(versions, key=lambda v: int(v.version))
    _model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    _model = mlflow.pyfunc.load_model(_model_uri)


@app.on_event("startup")
def startup_event() -> None:
    try:
        load_latest_model()
        print(f"[api] Loaded model: {_model_uri}")
    except Exception as e:
        print(f"[api] WARNING: model not loaded on startup: {e}")


# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_uri": _model_uri,
    }


@app.post("/reload")
def reload_model():
    """
    Reload the latest model from the registry (useful after retraining).
    """
    try:
        load_latest_model()
        return {"status": "reloaded", "model_uri": _model_uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame(req.rows)

        # PyFunc predict returns anomaly probability scores
        scores = _model.predict(df)

        # Normalize output
        if hasattr(scores, "values"):
            scores = scores.values

        scores_list = [float(x) for x in list(scores)]
        preds = [1 if s >= 0.5 else 0 for s in scores_list]

        return PredictResponse(
            model_uri=_model_uri,
            predictions=preds,
            scores=scores_list,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed. Check input schema. Error: {e}",
        )
