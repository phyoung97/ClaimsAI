import numbers
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# -----------------------------
# 1. CONFIG / CONSTANTS
# -----------------------------

# Default path to your processed, labeled dataset.
# Supports both .csv and .parquet via load_data().
DEFAULT_DATA_PATH = Path("data/processed/beneficiary_labeled.csv")

# Columns you definitely don't want as features (IDs, target, rule flags etc.)
DEFAULT_DROP_COLS = [
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

MLFLOW_EXPERIMENT_NAME = "claims_ai_anomaly_detection"

# Stable registered model names (used by API to load latest versions)
REGISTERED_XGB_MODEL_NAME = "claimsai_xgb_anomaly"
REGISTERED_ISO_MODEL_NAME = "claimsai_iso_anomaly"


# -----------------------------
# 2. DATA LOADING & SPLITTING
# -----------------------------


def load_data(data_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the processed, labeled dataset.
    Supports .parquet and .csv based on file suffix.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {data_path}\n"
            f"Update DEFAULT_DATA_PATH in train_model.py or pass a custom path."
        )

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")

    print(f"[load_data] Loaded shape: {df.shape}")
    return df


def get_target_column(df: pd.DataFrame) -> str:
    """
    Decide which column to use as the target.
    Prefers 'is_anomaly' if present, otherwise falls back to 'y_rule'.
    """
    if "is_anomaly" in df.columns:
        return "is_anomaly"
    elif "y_rule" in df.columns:
        return "y_rule"
    else:
        raise KeyError(
            "No target column found. Expected one of: ['is_anomaly', 'y_rule']"
        )


def make_features_and_target(
    df: pd.DataFrame,
    drop_cols: Optional[list] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into X (features) and y (target).

    IMPORTANT: We do NOT one-hot encode here anymore.
    Encoding/imputation happens inside an sklearn Pipeline so it is persisted
    and reused identically during inference (API/serving).
    """
    df = df.copy()

    if target_col is None:
        target_col = get_target_column(df)

    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    # Ensure target is excluded from features
    if target_col not in drop_cols:
        drop_cols = drop_cols + [target_col]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    print(
        f"[make_features_and_target] Features (raw): {X.shape[1]} | Target: {target_col}"
    )
    return X, y


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/validation split so the anomaly rate is similar in both sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[train_val_split] Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


# -----------------------------
# 3. PREPROCESSING PIPELINE
# -----------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a persisted preprocessing step that:
    - imputes missing numeric values with median
    - imputes missing categorical with most frequent
    - one-hot encodes categoricals with handle_unknown='ignore' (critical for serving)
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


# -----------------------------
# 4. METRIC HELPER
# -----------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute core anomaly detection metrics.
    - y_score: continuous scores (probabilities or anomaly scores)
    - y_pred: hard class predictions (0/1)
    """
    metrics: Dict[str, Any] = {}

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    except Exception:
        metrics["pr_auc"] = np.nan

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        metrics["roc_auc"] = np.nan

    try:
        metrics["precision"] = precision_score(y_true, y_pred)
    except Exception:
        metrics["precision"] = np.nan

    try:
        metrics["recall"] = recall_score(y_true, y_pred)
    except Exception:
        metrics["recall"] = np.nan

    try:
        metrics["f1"] = f1_score(y_true, y_pred)
    except Exception:
        metrics["f1"] = np.nan

    try:
        # multi-line string; we won't log as numeric metric
        metrics["classification_report"] = classification_report(y_true, y_pred)
    except Exception:
        metrics["classification_report"] = np.nan

    return metrics


# -----------------------------
# 5. XGBOOST MODEL (PIPELINE)
# -----------------------------


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier using the labeled anomalies.
    Returns the trained sklearn Pipeline and metrics.
    """
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    preprocessor = build_preprocessor(X_train)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_metrics(y_val.values, y_score, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "y_score": y_score,
        "y_pred": y_pred,
    }


# -----------------------------
# 6. ISOLATION FOREST MODEL (PIPELINE)
# -----------------------------


def train_isolation_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    contamination: float = 0.05,
) -> Dict[str, Any]:
    """
    Train an Isolation Forest (unsupervised) and evaluate it
    against your labeled anomalies for comparison.

    We persist preprocessing in the pipeline.
    """
    preprocessor = build_preprocessor(X_train)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", iso),
        ]
    )

    # IsolationForest ignores y; fit on X only
    model.fit(X_train)

    # decision_function needs transformed matrix
    X_val_t = model.named_steps["preprocess"].transform(X_val)

    # decision_function: higher = more normal; flip sign so higher = more anomalous
    raw_scores_val = model.named_steps["model"].decision_function(X_val_t)
    anomaly_score = -raw_scores_val

    # predict: -1 = anomaly, 1 = normal
    iso_pred = model.named_steps["model"].predict(X_val_t)
    y_pred = np.where(iso_pred == -1, 1, 0)

    metrics = compute_metrics(y_val.values, anomaly_score, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "y_score": anomaly_score,
        "y_pred": y_pred,
    }


# -----------------------------
# 7. MLFLOW LOGGING (WITH SIGNATURE + REGISTRATION)
# -----------------------------


def log_model_to_mlflow(
    model,
    run_name: str,
    metrics: dict,
    params: dict,
    X_example: pd.DataFrame,
    registered_model_name: Optional[str] = None,
) -> None:
    """
    Log model, params, and metrics to MLflow.

    - Logs a serving-ready sklearn Pipeline (preprocessing + estimator).
    - Adds signature + input_example for safer inference.
    - Optionally registers the model under a stable name so an API can load latest.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        # log params
        if params:
            for k, v in params.items():
                if v is not None:
                    mlflow.log_param(k, v)

        # log numeric metrics only
        if metrics:
            for k, v in metrics.items():
                if v is None:
                    continue

                if isinstance(v, numbers.Number):
                    v_float = float(v)
                    if not np.isnan(v_float):
                        mlflow.log_metric(k, v_float)
                else:
                    # log strings (e.g., classification_report) as params (truncated)
                    if isinstance(v, str):
                        mlflow.log_param(k, v[:250])

        # infer signature
        signature = None
        try:
            if hasattr(model, "predict_proba"):
                example_output = model.predict_proba(X_example.head(5))[:, 1]
            else:
                example_output = model.predict(X_example.head(5))
            signature = infer_signature(X_example.head(5), example_output)
        except Exception as e:
            print(f"[mlflow] WARNING: could not infer signature: {e}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_example.head(5),
            registered_model_name=registered_model_name,
        )

        print(
            f"[mlflow] Logged run='{run_name}' | registered='{registered_model_name}' | "
            f"metrics_keys={list(metrics.keys()) if metrics else []}"
        )


# -----------------------------
# 8. MAIN TRAINING ENTRYPOINT
# -----------------------------


def train_both_models(
    data_path: Path = DEFAULT_DATA_PATH,
    contamination: float = 0.05,
):
    """
    Load data, create train/val split, train XGBoost and IsolationForest,
    and log both to MLflow (with stable registered model names).
    """
    # 1) Data
    df = load_data(data_path)
    target_col = get_target_column(df)
    X, y = make_features_and_target(df, target_col=target_col)
    print("\n[DEBUG] Model expects the following raw input columns:")
    for c in X.columns:
        print(" -", c)

    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # 2) XGBoost
    print("\n====== Training XGBoost (supervised) ======")
    xgb_result = train_xgboost_model(X_train, y_train, X_val, y_val)
    log_model_to_mlflow(
        model=xgb_result["model"],
        run_name="xgboost_anomaly_classifier_1.3.2026",
        metrics=xgb_result["metrics"],
        params={
            "model_type": "xgboost_classifier",
            "target_col": target_col,
            "n_features_raw": X_train.shape[1],
        },
        X_example=X_train,
        registered_model_name=REGISTERED_XGB_MODEL_NAME,
    )

    # 3) Isolation Forest
    print("\n====== Training Isolation Forest (unsupervised) ======")
    iso_result = train_isolation_forest_model(
        X_train,
        y_train,
        X_val,
        y_val,
        contamination=contamination,
    )
    log_model_to_mlflow(
        model=iso_result["model"],
        run_name="isolation_forest_anomaly_detector_1.3.2026",
        metrics=iso_result["metrics"],
        params={
            "model_type": "isolation_forest",
            "contamination": contamination,
            "target_col": target_col,
            "n_features_raw": X_train.shape[1],
        },
        X_example=X_train,
        registered_model_name=REGISTERED_ISO_MODEL_NAME,
    )

    print("\n[train_both_models] Done.")
    print("XGBoost metrics:", xgb_result["metrics"])
    print("IsolationForest metrics:", iso_result["metrics"])


if __name__ == "__main__":
    train_both_models()
