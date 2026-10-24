import httpx
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
N_CLIENTS = 2000
DRIFT = False                                                                       # a switch: True = modified traffic, False = normal traffic

clients = pd.read_parquet("data/production.parquet").sample(N_CLIENTS, random_state=42)

if DRIFT:
    clients["EXT_SOURCE_2"] = clients["EXT_SOURCE_2"] * 0.7                         # external scores that are 30% lower, such as during an economic crisis when credit bureau scores plummet
    clients["DAYS_BIRTH"] = (clients["DAYS_BIRTH"] + 5 * 365).clip(upper=-6575)     # customers who are 5 years younger, as if we were targeting a new customer base
                                                                                    # .clip(upper=-6575) limits the values at −6,575 at most, which is 18 years, the limit specified in schema.py
status_codes = []
with httpx.Client() as http:
    for _, row in clients.iterrows():
        payload = row.dropna().to_dict()
        response = http.post(API_URL, json=payload)
        status_codes.append(response.status_code)

print(pd.Series(status_codes).value_counts())