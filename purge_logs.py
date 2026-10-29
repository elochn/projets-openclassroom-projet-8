import json
from datetime import datetime, timedelta, timezone

RETENTION_DAYS = 90
LOG_FILE = "logs/api_calls.jsonl"

limit = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

with open(LOG_FILE) as f:
    lines = f.readlines()

kept = [line for line in lines if datetime.fromisoformat(json.loads(line)["timestamp"]) >= limit]

with open(LOG_FILE, "w") as f:
    f.writelines(kept)

print(f"Deleted {len(lines) - len(kept)} calls older than {RETENTION_DAYS} days, kept {len(kept)}")