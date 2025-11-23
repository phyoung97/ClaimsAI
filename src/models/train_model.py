import numbers
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
    drop_cols: list = None,
    target_col: str = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into X (features) and y (target).
    - drop_cols are removed from the feature set if present.
    - target_col is removed from X and returned separately as y.
    Also:
    - One-hot encodes any object/category columns so models
      only see numeric data.
    """
    df = df.copy()

    if target_col is None:
        target_col = get_target_column(df)

    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    # ensure target is included in drop_cols so it never leaks into features
    if target_col not in drop_cols:
        drop_cols = drop_cols + [target_col]

    # X = all columns minus drop_cols
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    # ---- handle categorical columns (e.g., Gender, Race) ----
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        print(
            f"[make_features_and_target] One-hot encoding categorical columns: {cat_cols}"
        )
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    print(f"[make_features_and_target] Features: {X.shape[1]} | Target: {target_col}")
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
# 3. METRIC HELPER
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
        # This is a multi-line string, we'll keep it in metrics but
        # we WON'T log it as a numeric metric in MLflow.
        metrics["classification_report"] = classification_report(y_true, y_pred)
    except Exception:
        metrics["classification_report"] = np.nan

    return metrics


# -----------------------------
# 4. XGBOOST MODEL
# -----------------------------


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier using the labeled anomalies.
    Returns the trained model and metrics.
    """
    # Rough imbalance handling: ratio of negatives/positives
    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    model = XGBClassifier(
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

    model.fit(X_train, y_train)

    # Probabilities for the positive class (anomaly = 1)
    y_score = model.predict_proba(X_val)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_metrics(y_val.values, y_score, y_pred)

    result = {
        "model": model,
        "metrics": metrics,
        "y_score": y_score,
        "y_pred": y_pred,
    }

    return result


# -----------------------------
# 5. ISOLATION FOREST MODEL
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
    """
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,  # expected proportion anomalies
        random_state=42,
        n_jobs=-1,
    )

    iso.fit(X_train)

    # decision_function: higher scores = more normal
    # we flip the sign so higher = more anomalous (align with "probability-ish" view)
    raw_scores_val = iso.decision_function(X_val)
    anomaly_score = -raw_scores_val

    # predict: -1 = anomaly, 1 = normal
    iso_pred = iso.predict(X_val)
    y_pred = np.where(iso_pred == -1, 1, 0)

    metrics = compute_metrics(y_val.values, anomaly_score, y_pred)

    result = {
        "model": iso,
        "metrics": metrics,
        "y_score": anomaly_score,
        "y_pred": y_pred,
    }

    return result


# -----------------------------
# 6. MLFLOW LOGGING
# -----------------------------


def log_model_to_mlflow(
    model,
    model_name: str,
    metrics: dict,
    params: dict,
    feature_names=None,
    model_type: str = "xgboost",
) -> None:
    """
    Log model, params, and metrics to MLflow.
    model_name is used as the run name.
    """
    # Make sure we're in the right experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
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

                # Only treat real numbers as metrics
                if isinstance(v, numbers.Number):
                    v_float = float(v)
                    if not np.isnan(v_float):
                        mlflow.log_metric(k, v_float)
                else:
                    # Optional: if you want, log non-numeric stuff as params
                    # e.g., classification_report string
                    if isinstance(v, str):
                        mlflow.log_param(k, v[:250])  # truncate if very long

        # Log the model artifact (works for sklearn-compatible models, including XGBClassifier)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[mlflow] Logged {model_name} with metrics: {metrics}")


# -----------------------------
# 7. MAIN TRAINING ENTRYPOINT
# -----------------------------


def train_both_models(
    data_path: Path = DEFAULT_DATA_PATH,
    contamination: float = 0.05,
):
    """
    Load data, create train/val split, train XGBoost and IsolationForest,
    and log both to MLflow.
    """
    # 1) Data
    df = load_data(data_path)
    target_col = get_target_column(df)
    X, y = make_features_and_target(df, target_col=target_col)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    # 2) XGBoost
    print("\n====== Training XGBoost (supervised) ======")
    xgb_result = train_xgboost_model(X_train, y_train, X_val, y_val)
    log_model_to_mlflow(
        model=xgb_result["model"],
        model_name="xgboost_anomaly_classifier",
        metrics=xgb_result["metrics"],
        params={
            "model_type": "xgboost_classifier",
            "target_col": target_col,
            "n_features": X_train.shape[1],
        },
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
        model_name="isolation_forest_anomaly_detector",
        metrics=iso_result["metrics"],
        params={
            "model_type": "isolation_forest",
            "contamination": contamination,
            "target_col": target_col,
            "n_features": X_train.shape[1],
        },
    )

    print("\n[train_both_models] Done.")
    print("XGBoost metrics:", xgb_result["metrics"])
    print("IsolationForest metrics:", iso_result["metrics"])


if __name__ == "__main__":
    train_both_models()
