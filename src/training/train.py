"""Model training module.

Trains Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
LightGBM and CatBoost with 5-fold cross-validation, tunes each with Optuna,
tracks every run (params/metrics/artifacts/model) in MLflow, and selects +
registers the best model by ROC AUC.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.preprocessing.preprocess import run_preprocessing
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective_logistic_regression(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }
    model = LogisticRegression(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


def _objective_random_forest(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


def _objective_gradient_boosting(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
    }
    model = GradientBoostingClassifier(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


def _objective_xgboost(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


def _objective_lightgbm(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


def _objective_catboost(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 100, 400),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_state": 42,
        "verbose": False,
    }
    model = CatBoostClassifier(**params)
    return float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))


MODEL_REGISTRY: dict[str, tuple[Callable, Callable]] = {
    "logistic_regression": (_objective_logistic_regression, LogisticRegression),
    "random_forest": (_objective_random_forest, RandomForestClassifier),
    "gradient_boosting": (_objective_gradient_boosting, GradientBoostingClassifier),
    "xgboost": (_objective_xgboost, XGBClassifier),
    "lightgbm": (_objective_lightgbm, LGBMClassifier),
    "catboost": (_objective_catboost, CatBoostClassifier),
}


class ModelTrainer:
    """Orchestrates CV + Optuna tuning + MLflow tracking across the model zoo."""

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_config()
        mlflow.set_tracking_uri(self.cfg.training.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.training.mlflow_experiment_name)

    def _tune_model(self, name: str, X: np.ndarray, y: np.ndarray) -> tuple[Any, dict[str, Any], float]:
        objective_fn, model_cls = MODEL_REGISTRY[name]
        cv = StratifiedKFold(n_splits=self.cfg.training.cv_folds, shuffle=True, random_state=42)

        logger.info(f"Tuning {name} with Optuna ({self.cfg.training.optuna_trials} trials)")
        study = optuna.create_study(direction="maximize", study_name=f"{name}_study")
        study.optimize(
            lambda trial: objective_fn(trial, X, y, cv),
            n_trials=self.cfg.training.optuna_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"{name}: best CV ROC AUC={best_score:.4f}, params={best_params}")

        final_params = dict(best_params)
        if name in ("logistic_regression",):
            final_params.update({"max_iter": 1000, "random_state": 42})
        elif name == "xgboost":
            final_params.update({"random_state": 42, "n_jobs": -1, "eval_metric": "logloss"})
        elif name == "lightgbm":
            final_params.update({"random_state": 42, "n_jobs": -1, "verbosity": -1})
        elif name == "catboost":
            final_params.update({"random_state": 42, "verbose": False})
        else:
            final_params.update({"random_state": 42})

        model = model_cls(**final_params)
        model.fit(X, y)
        return model, best_params, best_score

    def train_all(self) -> dict[str, Any]:
        X_train, X_test, y_train, y_test, feature_names = run_preprocessing(self.cfg)

        results = {}
        best_model_name, best_model, best_test_auc = None, None, -1.0

        for name in self.cfg.training.models:
            with mlflow.start_run(run_name=name):
                model, best_params, cv_auc = self._tune_model(name, X_train, y_train)

                y_proba = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_proba)

                mlflow.log_params(best_params)
                mlflow.log_metric("cv_roc_auc", cv_auc)
                mlflow.log_metric("test_roc_auc", test_auc)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

                results[name] = {"cv_roc_auc": cv_auc, "test_roc_auc": test_auc, "params": best_params}
                logger.info(f"{name}: test ROC AUC={test_auc:.4f}")

                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_model = model
                    best_model_name = name

        logger.info(f"Best model: {best_model_name} with test ROC AUC={best_test_auc:.4f}")

        model_path = resolve_path(self.cfg.training.model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, model_path)

        metadata = {
            "best_model_name": best_model_name,
            "best_test_roc_auc": best_test_auc,
            "feature_names": feature_names,
            "trained_at": datetime.now(UTC).isoformat(),
            "all_results": {
                k: {"cv_roc_auc": v["cv_roc_auc"], "test_roc_auc": v["test_roc_auc"]} for k, v in results.items()
            },
            "model_version": self.cfg.project.version,
        }
        metadata_path = resolve_path(self.cfg.training.metadata_output_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved best model to {model_path} and metadata to {metadata_path}")

        with mlflow.start_run(run_name=f"{best_model_name}_registered"):
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=self.cfg.training.model_registry_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

        return {
            "best_model_name": best_model_name,
            "best_test_auc": best_test_auc,
            "results": results,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
        }


def main() -> None:
    trainer = ModelTrainer()
    trainer.train_all()


if __name__ == "__main__":
    main()
