"""Exploratory Data Analysis report generator.

Produces correlation heatmap, distribution plots, missing-value heatmap,
class imbalance plot, pairplots, boxplots, violin plots and summary
statistics for the raw healthcare readmission dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")

NUMERIC_COLS = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "blood_glucose",
    "hba1c",
    "cholesterol",
    "number_of_medications",
    "previous_admissions",
    "length_of_stay",
    "emergency_visits_last_year",
    "chronic_disease_count",
]
TARGET = "readmitted_30_days"


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df[NUMERIC_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    n_cols = 4
    n_rows = -(-len(NUMERIC_COLS) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2563eb")
        axes[i].set_title(col)
    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Distributions", y=1.02, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missing_value_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", ax=ax, yticklabels=False)
    ax.set_title("Missing Value Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "missing_value_heatmap.png", dpi=150)
    plt.close(fig)


def plot_class_imbalance(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No Readmit (0)", "Readmit (1)"], counts.values, color=["#2563eb", "#dc2626"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v} ({v / len(df):.1%})", ha="center", fontweight="bold")
    ax.set_title("Class Imbalance: 30-Day Readmission")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "class_imbalance.png", dpi=150)
    plt.close(fig)


def plot_pairplot(df: pd.DataFrame, output_dir: Path) -> None:
    subset_cols = ["age", "bmi", "hba1c", "previous_admissions", "length_of_stay", TARGET]
    clean = df[subset_cols].dropna()
    sample = clean.sample(n=min(800, len(clean)), random_state=42)
    g = sns.pairplot(
        sample, hue=TARGET, palette={0: "#2563eb", 1: "#dc2626"}, diag_kind="kde", plot_kws={"alpha": 0.5, "s": 15}
    )
    g.fig.suptitle("Pairplot of Key Features by Readmission Status", y=1.02)
    g.savefig(output_dir / "pairplot.png", dpi=150)
    plt.close(g.fig)


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    cols = ["previous_admissions", "length_of_stay", "hba1c", "chronic_disease_count"]
    fig, axes = plt.subplots(1, len(cols), figsize=(20, 5))
    for i, col in enumerate(cols):
        sns.boxplot(
            x=TARGET, y=col, data=df, hue=TARGET, ax=axes[i], palette={0: "#2563eb", 1: "#dc2626"}, legend=False
        )
        axes[i].set_title(f"{col} by Readmission")
        axes[i].set_xticklabels(["No", "Yes"])
    fig.tight_layout()
    fig.savefig(output_dir / "boxplots.png", dpi=150)
    plt.close(fig)


def plot_violin_plots(df: pd.DataFrame, output_dir: Path) -> None:
    cols = ["blood_glucose", "systolic_bp", "number_of_medications", "emergency_visits_last_year"]
    fig, axes = plt.subplots(1, len(cols), figsize=(20, 5))
    for i, col in enumerate(cols):
        sns.violinplot(
            x=TARGET, y=col, data=df, hue=TARGET, ax=axes[i], palette={0: "#2563eb", 1: "#dc2626"}, legend=False
        )
        axes[i].set_title(f"{col} by Readmission")
        axes[i].set_xticklabels(["No", "Yes"])
    fig.tight_layout()
    fig.savefig(output_dir / "violin_plots.png", dpi=150)
    plt.close(fig)


def generate_summary_statistics(df: pd.DataFrame, output_dir: Path) -> None:
    summary = df.describe(include="all").transpose()
    summary.to_csv(output_dir / "summary_statistics.csv")
    logger.info("Saved summary statistics CSV")


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating EDA visualizations")
    plot_correlation_heatmap(df, output_dir)
    plot_distributions(df, output_dir)
    plot_missing_value_heatmap(df, output_dir)
    plot_class_imbalance(df, output_dir)
    plot_pairplot(df, output_dir)
    plot_boxplots(df, output_dir)
    plot_violin_plots(df, output_dir)
    generate_summary_statistics(df, output_dir)
    logger.info(f"EDA complete. All visualizations saved to {output_dir}")


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(resolve_path(cfg.data.raw_path))
    output_dir = resolve_path("reports/eda")
    run_eda(df, output_dir)


if __name__ == "__main__":
    main()
