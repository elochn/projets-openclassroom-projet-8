import os
import time

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import warnings

# The old code path is measured as it was; its feature-name warning is a known false alarm (see README)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

os.environ["API_KEY"] = "benchmark-key"
from app import app, preprocess, state

N_TIMED = 300

pipe = joblib.load("model/model.joblib")
features = open("model/features.txt").read().splitlines()
clients = pd.read_parquet("data/production.parquet")
payloads = [row.dropna().to_dict() for _, row in clients.iterrows()]


def before(payload):
    return pipe.predict_proba(pd.DataFrame([payload], columns=features))[0, 1]


def after(payload):
    return state["booster"].predict(preprocess(payload))[0]


def ms_per_client(predict):
    start = time.perf_counter()
    for payload in payloads[:N_TIMED]:
        predict(payload)
    return (time.perf_counter() - start) / N_TIMED * 1000


with TestClient(app):
    proba_before = pipe.predict_proba(pd.DataFrame(payloads, columns=features))[:, 1]
    proba_after = np.array([after(payload) for payload in payloads])
    changed = ((proba_before >= 0.54) != (proba_after >= 0.54)).sum()
    print(f"Clients compared: {len(payloads)}")
    print(f"Largest probability difference: {np.abs(proba_before - proba_after).max()}")
    print(f"Decisions changed: {changed}")

    ms_before, ms_after = ms_per_client(before), ms_per_client(after)
    print(f"Before: {ms_before:.3f} ms per client")
    print(f"After:  {ms_after:.3f} ms per client ({ms_before / ms_after:.0f} times faster)")