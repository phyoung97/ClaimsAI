import mlflow
import numpy as np
import pandas as pd
import shap

REGISTERED_MODEL_NAME = "claimsai_xgb_anomaly"

# ----------------------------------------
# 1. Load model pipeline
# ----------------------------------------
model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
pipeline = mlflow.sklearn.load_model(model_uri)

preprocess = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# ----------------------------------------
# 2. Load data
# ----------------------------------------
df = pd.read_csv("data/processed/beneficiary_labeled.csv")

drop_cols = [
    "is_anomaly",
    "y_rule",
    "beneficiary_id",
    "rule_R_high_cost",
    "rule_R_high_out_ratio",
    "rule_R_high_chronic",
    "avg_reimb",
    "op_ratio",
    "car_ratio",
]

X = df[[c for c in df.columns if c not in drop_cols]]

# Sample rows for global explanation (adjust size as needed)
X_sample = X.sample(500, random_state=42)

X_background = X.sample(200, random_state=0)

X_sample_t = preprocess.transform(X_sample)
X_background_t = preprocess.transform(X_background)

feature_names = preprocess.get_feature_names_out()


# ----------------------------------------
# 3. Prediction function (class 1 prob)
# ----------------------------------------
def predict_proba_class1(X_t):
    return model.predict_proba(X_t)[:, 1]


# ----------------------------------------
# 4. SHAP explainer (model-agnostic)
# ----------------------------------------
masker = shap.maskers.Independent(X_background_t)

explainer = shap.Explainer(
    predict_proba_class1,
    masker=masker,
    feature_names=feature_names,
)

# ----------------------------------------
# 5. Compute SHAP values
# ----------------------------------------
explanation = explainer(X_sample_t)

# SHAP values: (n_rows, n_features)
shap_vals = explanation.values

# ----------------------------------------
# 6. Aggregate global importance
# ----------------------------------------
global_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(shap_vals), axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)

print("\nTop 20 Global SHAP Features:")
print(global_importance.head(20).to_string(index=False))
