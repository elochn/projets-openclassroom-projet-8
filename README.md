# Credit Scoring API — Prêt à Dépenser

Deployment and monitoring of the credit scoring model built in the
"Initiez-vous au MLOps" project (OpenClassrooms, part 1).

## Model

LightGBM classifier inside a scikit-learn Pipeline (median imputation with
missing-value indicators, scaling, one-hot encoding), trained on the Home Credit
dataset. It uses the 60 most important features selected with SHAP
(`credit_scoring_model` version 8 in the MLflow registry).

The full 422-feature model has the lowest business cost per client on the validation set
(0.4909 per client, with false negatives weighted 10 times more than false positives). 
The 60-feature model is deployed instead: it costs 0.7 % more (0.4943 per client) for 
7 times fewer input columns, which makes the API contract, the tests and the drift 
monitoring far simpler. Decision threshold: 0.54.

The API expects these 60 features (52 numeric, 8 categorical) already computed by
the upstream feature pipeline (client-level aggregations of the 7 Home Credit tables).

## Project structure

    model/        trained pipeline, feature list, parameters and metrics
    notebooks/    training pipeline (exploration, aggregation, modelling, SHAP)

## Live API

The API is deployed on Render (free instance):
https://projets-openclassroom-projet-8.onrender.com/docs

The free instance sleeps after 15 minutes without requests; the first call
after that takes about one minute.

## Run the API locally

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python -m uvicorn app:app --reload

Interactive documentation (Swagger): http://127.0.0.1:8000/docs

## Run the API with Docker

    docker build -t credit-scoring-api .
    docker run -p 7860:7860 credit-scoring-api

Interactive documentation (Swagger): http://127.0.0.1:7860/docs

## Example request

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "DAYS_BIRTH": -15000}'

`AMT_CREDIT`, `AMT_ANNUITY` and `DAYS_BIRTH` are required; the other 57
features are optional and imputed by the pipeline when missing. Invalid inputs
(missing required field, out-of-range value, wrong type, unknown category)
are rejected with a 422 response that names the field and the reason.

## Tests

    python -m pytest tests/ -v

## CI/CD

On every push to `main`, GitHub Actions runs three jobs in sequence :

1. `test`: installs the requirements and runs pytest.
2. `build`: builds the Docker image, starts a container and calls the API.
3. `deploy`: triggers a new deployment on Render through a deploy hook
   stored as the `RENDER_DEPLOY_HOOK_URL` repository secret.

## Status

Done: API, input validation, tests, Docker image, CI/CD pipeline, deployment.
In progress: storage of API calls and monitoring.