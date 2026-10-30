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

    app.py                       the API (FastAPI)
    schema.py                    input validation rules (pydantic)
    model/                       trained pipeline, feature list, parameters and metrics
    tests/                       API and model tests (pytest)
    Dockerfile                   container image of the API
    .github/workflows/ci.yml     CI/CD pipeline
    simulate_traffic.py          sends production clients to the API
    drift_report.py              data drift reports (Evidently)
    ops_report.py                error rate and latency
    dashboard.py                 monitoring dashboard (Streamlit)
    purge_logs.py                deletes logs older than 90 days
    screenshots/                 storage, drift reports and dashboard
    notebooks/                   training pipeline (exploration, aggregation, modelling, SHAP)

## Live API

The API is deployed on Render (free instance):
https://projets-openclassroom-projet-8.onrender.com/docs

The free instance sleeps after 15 minutes without requests; the first call
after that takes about one minute.

## Authentication

`/predict` requires an API key sent in the `X-API-Key` header; a missing or wrong
key returns 401. The API reads the expected key from the `API_KEY` environment
variable at startup and refuses to start without it. The production key is stored
as an environment variable on Render and is never committed. In Swagger, click
**Authorize** and paste the key.

## Run the API locally

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    export API_KEY=dev-key
    python -m uvicorn app:app --no-access-log

`dev-key` is any value you choose for local tests.
Interactive documentation (Swagger): http://127.0.0.1:8000/docs

## Run the API with Docker

    docker build -t credit-scoring-api .
    docker run -p 7860:7860 -e API_KEY=dev-key credit-scoring-api

Interactive documentation (Swagger): http://127.0.0.1:7860/docs

## Example request

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -H "X-API-Key: dev-key" \
      -d '{"AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "DAYS_BIRTH": -15000}'

`AMT_CREDIT`, `AMT_ANNUITY` and `DAYS_BIRTH` are required; the other 57
features are optional and imputed by the pipeline when missing. Invalid inputs
(missing required field, out-of-range value, wrong type, unknown category)
are rejected with a 422 response that names the field and the reason.

## Tests

    python -m pytest tests/ -v

## CI/CD

On every push to `main`, GitHub Actions runs three jobs in sequence:

1. `test`: installs the requirements and runs pytest.
2. `build`: builds the Docker image, starts a container with a test API key and
   calls the API.
3. `deploy`: triggers a new deployment on Render through a deploy hook
   stored as the `RENDER_DEPLOY_HOOK_URL` repository secret.

## Monitoring

### Storage

Every call to `/predict` is written as one JSON line to `logs/api_calls.jsonl`
(and to the terminal): status code, inputs, probability, decision, latency in
milliseconds and timestamp. Rejected requests (422) store only the faulty field
and the reason. This is a local proof of concept: the API runs locally, its logs
are stored and analysed on the same machine.

### How the monitoring data was produced

There are no real users, so production traffic is simulated:

- `data/reference.parquet`: the 307 511 training clients (60 features), the drift
  reference.
- `data/production.parquet`: the 48 744 clients of `application_test`, never seen
  during training.

Both are exported from the training notebook and are not committed (personal data).
`simulate_traffic.py` sends 2 000 random production clients to the API. With
`DRIFT = True`, it simulates a drift: `EXT_SOURCE_2` is multiplied by 0.7 and
clients are 5 years younger. The logs analysed below contain 2 000 normal calls
(window 1) followed by 2 000 drifted calls (window 2).

### Run the analysis

    pip install -r requirements-monitoring.txt
    python ops_report.py          # error rate, latency, faulty fields
    python drift_report.py        # one Evidently HTML report per window of 2 000 calls, in reports/
    streamlit run dashboard.py    # dashboard at http://localhost:8501
    python purge_logs.py          # deletes calls older than 90 days (run with the API stopped)

### How to read the drift report

Each column of the current window is compared with the training data:
normalised Wasserstein distance for numeric columns, Jensen-Shannon distance for
categorical ones. A column drifts when its score is at least 0.1. Evidently flags
a drift of the whole dataset when at least 50 % of the columns drift.

## Drift study

| | Window 1 (normal) | Window 2 (simulated drift) |
|---|---|---|
| Drifted columns | 8 / 60 | 10 / 60 |
| `EXT_SOURCE_2` score | 0.050 | 0.806 |
| `DAYS_BIRTH` score | 0.061 | 0.358 |
| Refusal rate | 26.1 % | 34.2 % |

**Window 1 already drifts on 8 columns without any simulated change.** The
production clients differ from the training clients: far fewer revolving loans
(9.5 % in training, 0.9 % in production), smaller credits (median `AMT_CREDIT`
513 531 vs 450 000) and therefore a higher `PAYMENT_RATE` (median 0.050 vs 0.061,
score 0.558). The refusal rate (26.1 %) is still the one expected from the
validation set (26.5 %).

**Window 2 detects the simulated drift** on exactly the two modified columns,
and the refusal rate rises from 26 % to 34 %: a lower external score makes
clients look riskier to the model.

### Points of vigilance

- **The global drift flag is not enough.** Window 2 changes one decision out of
  twelve, but only 10 of 60 columns drift, far below the 50 % needed to flag the
  dataset. Watch the most important features and the refusal rate individually.
- **Natural drift raises alerts on its own.** 8 columns drift with no simulated
  change; alerting on every column would create false alarms. The first windows
  of real traffic should be used as a baseline.
- **Drift is not always a problem.** A change of population (fewer revolving
  loans) can be legitimate; retraining is needed only if the performance drops,
  which requires the true outcomes (defaults), known months later.
- **Error rate and latency.** 0.1 % of errors (4 calls with a missing
  `AMT_ANNUITY`, a data quality issue, not an API bug). Median latency 3.9 ms,
  p95 5.8 ms, max 55.3 ms on the very first call (warm-up). `latency_ms` measures
  the model only, not the network; 401 responses are not logged.
- **Storage and cost.** One call takes about 2 KB (8.2 MB for 4 000 calls).
  At 10 000 calls a day, 90 days of logs take about 1.8 GB. On Render, the disk of
  the container is erased at every deployment, so a real deployment needs
  external storage (a database). Each drift report against the full reference
  takes about 30 seconds.

## GDPR

The logs contain no name or client ID, but age, gender and amounts together
identify most clients: they are pseudonymised personal data, still covered by
the GDPR. Measures taken:

- **Minimisation:** age rounded to the year and amounts to the thousand before
  logging (the drift scores do not change: 8 and 10 drifted columns before and
  after rounding); no client data for rejected requests; no IP addresses
  (`--no-access-log`).
- **Limited retention:** `purge_logs.py` deletes logs older than 90 days, enough
  to compare three months of traffic. In production, it would run every night.
- **Security:** API key; logs and data are never committed (`.gitignore`).
- **Hosting:** Render can keep its own connection logs, outside the API's control
  (see its privacy policy).
- **Automated decision (article 22):** a credit decision made by a model alone
  gives the client the right to a human review; the score should support a
  credit officer, not replace them.

## Status

Done: API, input validation, authentication, tests, Docker image, CI/CD pipeline,
deployment, logging, drift and operational monitoring, dashboard.
In progress: profiling and optimisation.