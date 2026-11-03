import os
import time
import warnings

import joblib
import numpy as np
import onnxruntime as rt
import pandas as pd
from fastapi.testclient import TestClient
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklFloat, StringTensorType

# The old code path is measured as it was; its feature-name warning is a known false alarm (see README)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

os.environ["API_KEY"] = "benchmark-key"
from app import app, preprocess, state

N_TIMED = 300

pipe = joblib.load("model/model.joblib")
features = open("model/features.txt").read().splitlines()
clients = pd.read_parquet("data/production.parquet")
payloads = [row.dropna().to_dict() for _, row in clients.iterrows()]

# 1. Convert the whole pipeline
categorical = pipe.named_steps["preprocessing"].transformers_[1][2]
inputs = [(c, StringTensorType([None, 1]) if c in categorical else SklFloat([None, 1])) for c in features]
try:
    convert_sklearn(pipe, initial_types=inputs)
except Exception as error:
    print("Test 1 - preprocessing + model in ONNX: failed:", str(error)[:120])

# 2. Convert LightGBM only, keep the numpy preprocessing
with TestClient(app):
    n_columns = preprocess(payloads[0]).shape[1]
    onnx_model = convert_lightgbm(state["booster"], initial_types=[("input", FloatTensorType([None, n_columns]))], zipmap=False, target_opset=15)
    session = rt.InferenceSession(onnx_model.SerializeToString())

    def numpy_booster(payload):
        return state["booster"].predict(preprocess(payload))[0]

    def numpy_onnx(payload):
        return session.run(None, {"input": preprocess(payload).astype(np.float32)})[1][0, 1]

    proba_pipeline = pipe.predict_proba(pd.DataFrame(payloads, columns=features))[:, 1]
    proba_onnx = np.array([numpy_onnx(payload) for payload in payloads])
    changed = ((proba_pipeline >= 0.54) != (proba_onnx >= 0.54)).sum()
    print("Test 2 - preprocessing in numpy, model in ONNX:")
    print(f"  largest probability difference: {np.abs(proba_pipeline - proba_onnx).max():.4f}")
    print(f"  decisions changed: {changed} / {len(payloads)}")

    for name, predict in [("  model in LightGBM (deployed)", numpy_booster), ("  model in ONNX Runtime", numpy_onnx)]:
        start = time.perf_counter()
        for payload in payloads[:N_TIMED]:
            predict(payload)
        print(f"{name}: {(time.perf_counter() - start) / N_TIMED * 1000:.3f} ms per client")