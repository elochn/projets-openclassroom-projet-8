import pytest
from fastapi.testclient import TestClient # a mock browser provided by FastAPI. It sends requests to your API without running Uvicorn.

from app import app 


def test_health_returns_model_info():
    with TestClient(app) as client:
        # Same as opening http://127.0.0.1:8000/ in a browser
        response = client.get("/")

    assert response.status_code == 200
    print(response.status_code)

    body = response.json() # The content of the answer, converted from JSON to a Python dictionary
    print(body)

    assert body["n_features"] == 60
    print(body["n_features"])

    assert body["threshold"] == 0.54


def test_predict_returns_score_and_decision():
    payload = {"AMT_CREDIT": 500000.0, "AMT_ANNUITY": 25000.0, "DAYS_BIRTH": -15000, "CODE_GENDER": "F"}

    with TestClient(app) as client:
        response = client.post(
            "/predict", json=payload # equivalent to the “Execute” button in Swagger, 
            )                        # with `payload` as the request body

    assert response.status_code == 200
    body = response.json()
    assert body["probability_default"] == pytest.approx(0.4023, abs=1e-4)
    assert body["decision"] == "accepted"


def test_missing_required_field_is_rejected():
    # missing required field : AMT_ANNUITY
    payload = {"AMT_CREDIT": 500000.0, "DAYS_BIRTH": -15000}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "AMT_ANNUITY"]
    assert error["msg"] == "Field required"


def test_negative_age_is_rejected():
    # DAYS_BIRTH counts client's age in days before the application, so it must be negative
    payload = {"AMT_CREDIT": 500000.0, "AMT_ANNUITY": 25000.0, "DAYS_BIRTH": 15000}  

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "DAYS_BIRTH"]


def test_zero_credit_is_rejected():
    payload = {"AMT_CREDIT": 0, "AMT_ANNUITY": 25000.0, "DAYS_BIRTH": -15000}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "AMT_CREDIT"]


def test_text_instead_of_number_is_rejected():
    payload = {"AMT_CREDIT": 500000.0, "AMT_ANNUITY": 25000.0, "DAYS_BIRTH": "abc"}

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "DAYS_BIRTH"]