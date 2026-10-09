from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI

# The class that describes and validates the request body
from schema import ClientData

# Filled once at startup, then read by every request
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once, when the server starts
    state["pipe"] = joblib.load("model/model.joblib")
    with open("model/features.txt") as f:
        state["features"] = f.read().splitlines()
    metrics = pd.read_csv("model/metrics.csv")
    state["threshold"] = round(float(metrics.loc[0, "best_threshold"]), 2)
    yield
    # Runs once, when the server stops
    state.clear()


app = FastAPI(title="Credit Scoring API", lifespan=lifespan)

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

@app.post("/predict")
def predict(client: ClientData):
    values = client.model_dump(exclude_none=True)
    X = pd.DataFrame([values], columns=state["features"])
    proba = float(state["pipe"].predict_proba(X)[0, 1])
    decision = "refused" if proba >= state["threshold"] else "accepted"
    return {
        "probability_default": round(proba, 4),
        "threshold": state["threshold"],
        "decision": decision,
    }