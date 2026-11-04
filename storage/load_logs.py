import json
import os

import psycopg

LOG_FILE = "logs/api_calls.jsonl"


def as_json(value):
    return None if value is None else json.dumps(value)


with open(LOG_FILE) as f:
    records = [json.loads(line) for line in f]

rows = [
    (
        record["timestamp"],
        record["status_code"],
        as_json(record.get("inputs")),
        as_json(record.get("errors")),
        record.get("probability_default"),
        record.get("decision"),
        record.get("latency_ms"),
    )
    for record in records
]

with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
    with open("storage/schema.sql") as f:
        connection.execute(f.read())
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO api_calls
                (called_at, status_code, inputs, errors, probability_default, decision, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (called_at) DO NOTHING
            """,
            rows,
        )
    total = connection.execute("SELECT count(*) FROM api_calls").fetchone()[0]

print(f"Read {len(rows)} calls from {LOG_FILE}; the table now holds {total} calls")