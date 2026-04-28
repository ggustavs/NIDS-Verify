"""EDA plots for extracted NIDS features."""

import argparse
import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure

_BENIGN_COLOR = "#2E8B57"
_MALICIOUS_COLOR = "#DC143C"
_PALETTE = {"Benign": _BENIGN_COLOR, "Malicious": _MALICIOUS_COLOR}


def _label_names(s: pd.Series) -> pd.Series:
    return s.map(lambda v: "Benign" if v == 0 else "Malicious")


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {path}")


class DataVisualizer:
    def __init__(self, data_file: str, output_dir: str = "visualizations"):
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.df: pd.DataFrame | None = None
        sns.set_theme(style="whitegrid")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_file)
        if df.empty:
            raise ValueError(f"Empty dataset: {self.data_file}")
        if "Label" not in df.columns:
            df = df.rename(columns={df.columns[-1]: "Label"})
        self.df = df
        logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    def _data(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        assert self.df is not None
        return self.df

    def plot_label_distribution(self, save: bool = True) -> Figure:
        df = self._data()
        labels = _label_names(df["Label"])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=labels, palette=_PALETTE, ax=ax)
        ax.set(xlabel="Class", ylabel="Flows", title="Label distribution")
        for c in ax.containers:
            ax.bar_label(c, fmt="{:,.0f}")
        if save:
            _save(fig, self.output_dir / "label_distribution.png")
        return fig

    def plot_feature_histogram(self, feature: str, bins: int = 30, save: bool = True) -> Figure:
        df = self._data()
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found")
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data = df[[feature, "Label"]].assign(Label=_label_names(df["Label"]))
        sns.histplot(
            data=plot_data,
            x=feature,
            hue="Label",
            bins=bins,
            stat="density",
            palette=_PALETTE,
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature}")
        if save:
            _save(fig, self.output_dir / f"{feature}_histogram.png")
        return fig

    def plot_feature_scatter(
        self, x: str, y: str, sample_size: int | None = 5000, save: bool = True
    ) -> Figure:
        df = self._data()
        for col in (x, y):
            if col not in df.columns:
                raise ValueError(f"Feature '{col}' not found")
        plot_df = (
            df.sample(n=sample_size, random_state=42)
            if sample_size and len(df) > sample_size
            else df
        )
        plot_data = plot_df[[x, y, "Label"]].assign(Label=_label_names(plot_df["Label"]))
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(
            data=plot_data,
            x=x,
            y=y,
            hue="Label",
            palette=_PALETTE,
            alpha=0.6,
            s=20,
            ax=ax,
        )
        ax.set_title(f"{x} vs {y}")
        if save:
            _save(fig, self.output_dir / f"{x}_vs_{y}_scatter.png")
        return fig

    def plot_correlation_matrix(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        save: bool = True,
    ) -> Figure:
        df = self._data()
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, square=True, ax=ax)
        ax.set_title(f"Feature correlations ({method})")
        if save:
            _save(fig, self.output_dir / f"correlation_{method}.png")
        return fig

    def plot_label_correlations(self, top_n: int = 20, save: bool = True) -> Figure:
        """Per-feature |Pearson(feature, label)|. NOT model-based importance."""
        df = self._data()
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Label"]
        corrs = (
            df[numeric_cols + ["Label"]]
            .corr()["Label"]
            .drop("Label")
            .abs()
            .dropna()
            .sort_values(ascending=False)
            .head(top_n)
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.barplot(x=corrs.values, y=corrs.index, color="steelblue", ax=ax)
        ax.set(
            xlabel="|Pearson correlation with Label|",
            ylabel="Feature",
            title=f"Top {top_n} label-correlated features",
        )
        if save:
            _save(fig, self.output_dir / "label_correlations.png")
        return fig

    def generate_comprehensive_report(self, sample_features: list[str] | None = None) -> dict:
        df = self._data()
        plots: list[str] = []

        self.plot_label_distribution()
        plots.append("label_distribution.png")
        self.plot_label_correlations()
        plots.append("label_correlations.png")
        self.plot_correlation_matrix()
        plots.append("correlation_pearson.png")

        if sample_features is None:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c != "Label"
            ]
            sample_features = numeric_cols[:5]
        for feat in sample_features:
            self.plot_feature_histogram(feat)
            plots.append(f"{feat}_histogram.png")

        logger.info(f"Generated {len(plots)} plots in {self.output_dir}")
        return {
            "dataset_shape": df.shape,
            "output_directory": str(self.output_dir),
            "plots_generated": plots,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA visualisations for NIDS features")
    parser.add_argument("data_file")
    parser.add_argument("--output-dir", default="visualizations")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--histogram")
    parser.add_argument("--scatter", nargs=2, metavar=("X", "Y"))
    parser.add_argument("--correlation", action="store_true")
    parser.add_argument(
        "--importance",
        action="store_true",
        help="Plot |Pearson(feature, label)| (NOT model-based importance)",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    viz = DataVisualizer(args.data_file, args.output_dir)
    viz.load_data()

    if args.report:
        viz.generate_comprehensive_report()
        return 0

    plots = 0
    if args.histogram:
        viz.plot_feature_histogram(args.histogram)
        plots += 1
    if args.scatter:
        viz.plot_feature_scatter(args.scatter[0], args.scatter[1], args.sample_size)
        plots += 1
    if args.correlation:
        viz.plot_correlation_matrix()
        plots += 1
    if args.importance:
        viz.plot_label_correlations()
        plots += 1

    if plots == 0:
        logger.warning("No plot requested. Use --report for full analysis.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
