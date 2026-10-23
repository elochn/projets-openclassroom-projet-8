import httpx
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
N_CLIENTS = 2000

clients = pd.read_parquet("data/production.parquet").sample(N_CLIENTS, random_state=42)

status_codes = []
with httpx.Client() as http:
    for _, row in clients.iterrows():
        payload = row.dropna().to_dict()
        response = http.post(API_URL, json=payload)
        status_codes.append(response.status_code)

print(pd.Series(status_codes).value_counts())