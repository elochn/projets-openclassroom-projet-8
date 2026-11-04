-- One row per call to /predict, as written in logs/api_calls.jsonl
CREATE TABLE IF NOT EXISTS api_calls (
    id                  BIGSERIAL PRIMARY KEY,
    called_at           TIMESTAMPTZ NOT NULL UNIQUE,
    status_code         SMALLINT NOT NULL,
    inputs              JSONB,
    errors              JSONB,
    probability_default REAL,
    decision            TEXT CHECK (decision IN ('accepted', 'refused')),
    latency_ms          REAL
);

CREATE INDEX IF NOT EXISTS api_calls_called_at_idx ON api_calls (called_at);

-- Daily operational monitoring, computed from api_calls
CREATE OR REPLACE VIEW daily_monitoring AS
SELECT
    called_at::date AS day,
    count(*) AS calls,
    round(avg((status_code <> 200)::int) * 100, 2) AS error_rate_pct,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS median_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    round(avg((decision = 'refused')::int) * 100, 2) AS refusal_rate_pct
FROM api_calls
GROUP BY day
ORDER BY day;