import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader

# The class that describes and validates the request body
from schema import ClientData

# Filled once at startup, then read by every request
state = {}

# One JSON line per API call, written to a file and to the terminal
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("api_calls")
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler("logs/api_calls.jsonl"))
logger.addHandler(logging.StreamHandler())

def log_call(record):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.info(json.dumps(record))

def generalise(values):
    stored = dict(values)
    stored["DAYS_BIRTH"] = round(stored["DAYS_BIRTH"] / 365) * 365
    for column in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if column in stored:
            stored[column] = round(stored[column] / 1000) * 1000
    return stored


def preprocess(values):
    x = np.array([values.get(column, np.nan) for column in state["num_cols"]], dtype=float)
    missing = np.isnan(x)
    numeric = np.concatenate([np.where(missing, state["medians"], x), missing[state["flagged"]]])
    numeric = (numeric - state["mean"]) / state["scale"]
    onehot = [categories == values.get(column, "Unknown") for column, categories in zip(state["cat_cols"], state["categories"])]
    return np.concatenate([numeric, *onehot]).reshape(1, -1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, when the server starts
    pipe = joblib.load("model/model.joblib")
    preprocessing = pipe.named_steps["preprocessing"]
    numeric = preprocessing.named_transformers_["num"]
    categorical = preprocessing.named_transformers_["cat"]
    state["num_cols"] = preprocessing.transformers_[0][2]
    state["cat_cols"] = preprocessing.transformers_[1][2]
    state["medians"] = numeric.named_steps["imputer"].statistics_
    state["flagged"] = numeric.named_steps["imputer"].indicator_.features_
    state["mean"] = numeric.named_steps["scaler"].mean_
    state["scale"] = numeric.named_steps["scaler"].scale_
    state["categories"] = categorical.named_steps["encoder"].categories_
    state["booster"] = pipe.named_steps["model"].booster_

    with open("model/features.txt") as f:
        state["features"] = f.read().splitlines()
    metrics = pd.read_csv("model/metrics.csv")
    state["threshold"] = round(float(metrics.loc[0, "best_threshold"]), 2)
    state["api_key"] = os.environ["API_KEY"]
    yield
    # Runs once, when the server stops
    state.clear()


app = FastAPI(title="Credit Scoring API", lifespan=lifespan)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str | None = Security(api_key_header)):
    if key is None or not secrets.compare_digest(key, state["api_key"]):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# Called by FastAPI whenever a request body breaks a rule of ClientData
@app.exception_handler(RequestValidationError)
async def log_invalid_request(request: Request, exc: RequestValidationError):
    log_call({
        "status_code": 422,
        "errors": [{"field": e["loc"][-1], "message": e["msg"]} for e in exc.errors()],
    })
    return await request_validation_exception_handler(request, exc)


@app.get("/")
def health():
    return {
        "message": "The API is operational ✅",
        "n_features": len(state["features"]),
        "threshold": state["threshold"],
        }

@app.get("/features")
def features():
    return {
        "features": state["features"]
    }

@app.post(
        "/predict", 
        dependencies=[Depends(verify_api_key)], 
        responses={
            401: {"description": "Missing or invalid token"}, 
        }
)
def predict(client: ClientData):
    start = time.perf_counter()
    values = client.model_dump(exclude_none=True)
    X = preprocess(values)
    proba = float(state["booster"].predict(X)[0])
    decision = "refused" if proba >= state["threshold"] else "accepted"
    latency_ms = (time.perf_counter() - start) * 1000
    log_call({
        "status_code": 200,
        "inputs": generalise(values),
        "probability_default": round(proba, 4),
        "decision": decision,
        "latency_ms": round(latency_ms, 2),
        })
    return {
        "probability_default": round(proba, 4),
        "threshold": state["threshold"],
        "decision": decision,
    }