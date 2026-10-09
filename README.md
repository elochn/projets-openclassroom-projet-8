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

## Status

Work in progress: API, tests, Docker image, CI/CD pipeline and monitoring are
added step by step.