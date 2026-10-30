import pandas as pd
import streamlit as st
from evidently import Report
from evidently.presets import DataDriftPreset

WINDOW = 2000

st.title("Credit scoring API monitoring")

logs = pd.read_json("logs/api_calls.jsonl", lines=True)
logs["window"] = logs.index // WINDOW + 1
successes = logs[logs["status_code"] == 200]

st.header("Operations")
calls, errors, median, p95 = st.columns(4)
calls.metric("Calls", len(logs))
errors.metric("Error rate", f"{(logs['status_code'] != 200).mean():.1%}")
median.metric("Median latency", f"{successes['latency_ms'].median():.1f} ms")
p95.metric("p95 latency", f"{successes['latency_ms'].quantile(0.95):.1f} ms")

st.subheader("Latency of each call (ms)")
st.line_chart(successes["latency_ms"].reset_index(drop=True))

st.subheader("Refusal rate per window" )
st.bar_chart(successes.groupby("window")["decision"].apply(lambda d: (d == "refused").mean()))


@st.cache_data
def drift_scores(window):
    features = open("model/features.txt").read().splitlines()
    reference = pd.read_parquet("data/reference.parquet")
    current = pd.json_normalize(successes[successes["window"] == window]["inputs"]).reindex(columns=features)
    result = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference).dict()
    rows = [
        {"column": m["config"]["column"], "method": m["config"]["method"], "score": m["value"], "threshold": m["config"]["threshold"]}
        for m in result["metrics"]
        if m["metric_name"].startswith("ValueDrift")
    ]
    return pd.DataFrame(rows).sort_values("score", ascending=False)


st.header("Data drift against the training data")
window = st.selectbox("Window", sorted(logs["window"].unique()))
scores = drift_scores(window)
drifted = scores[scores["score"] >= scores["threshold"]]
st.metric("Drifted columns", f"{len(drifted)} / {len(scores)}")
st.dataframe(drifted, hide_index=True)