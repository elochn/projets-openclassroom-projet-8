# Credit Scoring API — Prêt à Dépenser

Deployment and monitoring of the credit scoring model built in the
"Initiez-vous au MLOps" project (OpenClassrooms, part 1).

## Model

LightGBM classifier inside a scikit-learn Pipeline (median imputation with
missing-value indicators, scaling, one-hot encoding), trained on the Home Credit
dataset. It is the champion of the MLflow registry (`credit_scoring_model`,
version 6): the lowest business cost per client on the validation set, with false
negatives weighted 10 times more than false positives. Decision threshold: 0.52.

The API expects the 422 model features already computed by the upstream feature
pipeline (client-level aggregations of the 7 Home Credit tables).

## Project structure

    model/        trained pipeline, feature list, parameters and metrics
    notebooks/    training pipeline (exploration, aggregation, modelling, SHAP)

## Status

Work in progress: API, tests, Docker image, CI/CD pipeline and monitoring are
added step by step.