import joblib
import pandas as pd
import pytest

def test_model_scores_one_client():
    # Load the three files the API will need
    pipe = joblib.load("model/model.joblib")
    with open("model/features.txt") as f:
        features = f.read().splitlines()
    metrics = pd.read_csv("model/metrics.csv")
    threshold = round(float(metrics.loc[0, "best_threshold"]), 2)

    # Same client as in notebooks/scratch_predict.py
    client = {
        "AMT_CREDIT": 500000.0,
        "DAYS_BIRTH": -15000,
        "CODE_GENDER": "F",
    }

    X = pd.DataFrame([client], columns=features)
    proba = float(pipe.predict_proba(X)[0, 1])

    assert len(features) == 60
    assert threshold == 0.54
    assert proba == pytest.approx(0.4023, abs=1e-4)
