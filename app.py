import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import joblib
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, when the server starts
    state["pipe"] = joblib.load("model/model.joblib")
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
        "inputs": exc.body,
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
    X = pd.DataFrame([values], columns=state["features"])
    proba = float(state["pipe"].predict_proba(X)[0, 1])
    decision = "refused" if proba >= state["threshold"] else "accepted"
    latency_ms = (time.perf_counter() - start) * 1000
    log_call({
        "status_code": 200,
        "inputs": values,
        "probability_default": round(proba, 4),
        "decision": decision,
        "latency_ms": round(latency_ms, 2),
        })
    return {
        "probability_default": round(proba, 4),
        "threshold": state["threshold"],
        "decision": decision,
    }