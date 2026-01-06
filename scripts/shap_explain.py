import mlflow
import numpy as np
import pandas as pd
import shap

REGISTERED_MODEL_NAME = "claimsai_xgb_anomaly"

# -------------------------------
# 1) Load the registered pipeline
# -------------------------------
model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
pipeline = mlflow.sklearn.load_model(model_uri)

preprocess = pipeline.named_steps["preprocess"]
xgb_model = pipeline.named_steps["model"]

# -------------------------------
# 2) Load data + build background
# -------------------------------
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

# Use a small background sample (keeps it fast)
X_background = X.sample(200, random_state=42)
X_background_t = preprocess.transform(X_background)

# Feature names after preprocessing (num + one-hot)
feature_names = preprocess.get_feature_names_out()


# -------------------------------
# 3) Define a probability function on TRANSFORMED input
# -------------------------------
def predict_proba_class1(X_t):
    # X_t is the preprocessed matrix (numpy / scipy sparse)
    return xgb_model.predict_proba(X_t)[:, 1]


# -------------------------------
# 4) Build a SHAP explainer (model-agnostic)
# -------------------------------
# masker expects the same representation as X_t
masker = shap.maskers.Independent(X_background_t)

explainer = shap.Explainer(
    predict_proba_class1,
    masker=masker,
    feature_names=feature_names,
)

# -------------------------------
# 5) Explain ONE row (raw format)
# -------------------------------
row = pd.DataFrame(
    [
        {
            "AGE": 72,
            "Date_of_Death": 0,
            "Gender": "M",
            "Race": "1",
            "total_coverage_months": 12,
            "chronic_count": 5,
            "Number_of_months_covered_a": 12,
            "Numver_of_months_covered_b": 12,
            "Number_of_months_HMO_coverage": 0,
            "Number_of_months_covered_d": 12,
        }
    ]
)

row_t = preprocess.transform(row)

explanation = explainer(row_t)

# explanation.values: shape (n_rows, n_features)
vals = explanation.values[0]
base = float(explanation.base_values[0])

shap_df = (
    pd.DataFrame({"feature": feature_names, "shap_value": vals})
    .assign(abs_val=lambda d: np.abs(d["shap_value"]))
    .sort_values("abs_val", ascending=False)
    .drop(columns=["abs_val"])
)

print(f"\nModel URI: {model_uri}")
print(f"SHAP base value (expected prob): {base:.6f}")
print("\nTop SHAP contributors (by absolute impact):")
print(shap_df.head(15).to_string(index=False))

# Optional: sanity check predicted probability
pred_prob = float(predict_proba_class1(row_t)[0])
print(f"\nPredicted anomaly probability: {pred_prob:.6f}")
print(f"Predicted class (>=0.5): {1 if pred_prob >= 0.5 else 0}")
