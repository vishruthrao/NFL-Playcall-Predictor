import joblib
import matplotlib.pyplot as plt
from playcall import feature_importance

model = joblib.load("models/play_call_model.joblib")
imp = feature_importance(model, top_n=20).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(imp["feature"], imp["importance"])
plt.xlabel("XGBoost feature importance")
plt.title("NFL Play-Call Predictor: Top Features")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=160)
print("Saved outputs/feature_importance.png")
