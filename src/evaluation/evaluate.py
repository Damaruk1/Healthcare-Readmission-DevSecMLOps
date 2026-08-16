"""Model evaluation module.

Computes classification metrics (accuracy, precision, recall, F1, ROC AUC,
PR AUC), and generates confusion matrix, ROC curve, PR curve, calibration
curve, SHAP explainability, and feature importance plots for the best model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class ModelEvaluator:
    """Evaluates a trained model on held-out test data and produces reports/figures."""

    def __init__(self, model: Any, feature_names: list[str], output_dir: Path) -> None:
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        logger.info(
            f"Metrics: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}"
        )
        return metrics

    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"]).plot(ax=ax, cmap="Blues", colorbar=True)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(self.model, X_test, y_test, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(self.output_dir / "roc_curve.png", dpi=150)
        plt.close(fig)

    def plot_pr_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(self.model, X_test, y_test, ax=ax)
        ax.set_title("Precision-Recall Curve")
        fig.tight_layout()
        fig.savefig(self.output_dir / "pr_curve.png", dpi=150)
        plt.close(fig)

    def plot_calibration_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        y_proba = self.model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "calibration_curve.png", dpi=150)
        plt.close(fig)

    def plot_feature_importance(self) -> None:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            logger.warning("Model has no feature_importances_ attribute; skipping feature importance plot")
            return
        order = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(
            [self.feature_names[i] for i in order][::-1],
            importances[order][::-1],
            color="#2563eb",
        )
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(self.output_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    def plot_shap_summary(self, X_test: np.ndarray, sample_size: int = 200) -> None:
        try:
            sample = X_test[: min(sample_size, len(X_test))]
            explainer = shap.Explainer(self.model, sample)
            shap_values = explainer(sample)

            fig = plt.figure(figsize=(9, 8))
            if len(shap_values.shape) == 3:  # multi-class output, take positive class
                shap.summary_plot(shap_values[:, :, 1].values, sample, feature_names=self.feature_names, show=False)
            else:
                shap.summary_plot(shap_values.values, sample, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("SHAP summary plot saved")
        except Exception as exc:  # SHAP explainer support varies by model type
            logger.warning(f"SHAP explainability plot skipped due to: {exc}")

    def run_full_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        metrics = self.compute_metrics(X_test, y_test)
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_roc_curve(X_test, y_test)
        self.plot_pr_curve(X_test, y_test)
        self.plot_calibration_curve(X_test, y_test)
        self.plot_feature_importance()
        self.plot_shap_summary(X_test)

        metrics_path = self.output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {metrics_path} and figures to {self.output_dir}")
        return metrics


def main() -> None:
    cfg = load_config()
    model = joblib.load(resolve_path(cfg.training.model_output_path))

    test_df = pd.read_csv(resolve_path(cfg.data.processed_test_path))
    target_col = cfg.data.target_column
    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values
    feature_names = test_df.drop(columns=[target_col]).columns.tolist()

    output_dir = resolve_path("reports/evaluation")
    evaluator = ModelEvaluator(model, feature_names, output_dir)
    evaluator.run_full_evaluation(X_test, y_test)


if __name__ == "__main__":
    main()
