import cProfile
import logging
import os
import pstats

import pandas as pd
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "profile-key"
from app import app, logger

N_CLIENTS = 300

logger.handlers = [logging.NullHandler()]
clients = pd.read_parquet("data/production.parquet").sample(N_CLIENTS, random_state=0)
payloads = [row.dropna().to_dict() for _, row in clients.iterrows()]

with TestClient(app) as client:
    profiler = cProfile.Profile()
    profiler.enable()
    for payload in payloads:
        client.post("/predict", json=payload, headers={"X-API-Key": "profile-key"})
    profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative").print_stats("app.py|pipeline.py|_column_transformer.py|sklearn.py", 10)