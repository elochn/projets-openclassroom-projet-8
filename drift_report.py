import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

WINDOW = 2000

features = open("model/features.txt").read().splitlines()
reference = pd.read_parquet("data/reference.parquet")
logs = pd.read_json("logs/api_calls.jsonl", lines=True)

os.makedirs("reports", exist_ok=True)
for number, start in enumerate(range(0, len(logs), WINDOW), start=1):
    window = logs.iloc[start:start + WINDOW]
    scored = window[window["status_code"] == 200]
    current = pd.json_normalize(scored["inputs"]).reindex(columns=features)

    result = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    path = f"reports/data_drift_{number}.html"
    result.save_html(path)
    print("Saved", path)