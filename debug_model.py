# save as debug_model.py and run: python debug_model.py
import joblib, os
import numpy as np

MODEL_DIR = "models"

# Test lung SVM
path = os.path.join(MODEL_DIR, "lung_model.pkl")
obj = joblib.load(path)
print("Type:", type(obj))
print("Keys (if dict):", obj.keys() if isinstance(obj, dict) else "Not a dict")

pipeline = obj["model"] if isinstance(obj, dict) else obj
print("Has predict_proba:", hasattr(pipeline, "predict_proba"))

# Test feature shape
dummy = np.random.rand(1, 8164)
try:
    out = pipeline.predict(dummy)
    print("Predict OK:", out)
except Exception as e:
    print("Predict FAILED:", e)

try:
    proba = pipeline.predict_proba(dummy)
    print("Proba OK:", proba)
except Exception as e:
    print("Proba FAILED:", e)