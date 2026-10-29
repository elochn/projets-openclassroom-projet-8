import pandas as pd

logs = pd.read_json("logs/api_calls.jsonl", lines=True)
successes = logs[logs["status_code"] == 200]

print("Calls:", len(logs))
print(f"Error rate: {(logs['status_code'] != 200).mean():.1%}")
print(logs["status_code"].value_counts())

latency = successes["latency_ms"]
print(f"Latency (ms): median {latency.median():.1f}, p95 {latency.quantile(0.95):.1f}, max {latency.max():.1f}")

fields = logs["errors"].dropna().explode().map(lambda error: (error["field"], error["message"])) # which fields are causing these errors?
print(fields.value_counts())