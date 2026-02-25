"""
Data Visualization and Analysis Tools

Comprehensive visualization utilities for NIDS feature analysis,
including distribution plots, correlation analysis, and attack pattern visualization.
"""

import argparse
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.figure import Figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization tool for NIDS feature analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Generate full report
        python plotter.py features.csv --report

        # Specific visualizations
        python plotter.py features.csv --histogram Flow_Duration --scatter Flow_Duration Pkt_Size1

        # Custom output directory
        python plotter.py features.csv --report --output-dir ./my_plots
        """,
    )

    parser.add_argument("data_file", help="Path to the CSV file with extracted features")
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory to save visualizations (default: visualizations)",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate comprehensive visualization report"
    )
    parser.add_argument("--histogram", help="Generate histogram for specific feature")
    parser.add_argument(
        "--scatter", nargs=2, metavar=("X", "Y"), help="Generate scatter plot for two features"
    )
    parser.add_argument("--correlation", action="store_true", help="Generate correlation matrix")
    parser.add_argument(
        "--importance", action="store_true", help="Generate feature importance analysis"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Sample size for scatter plots (default: 5000)",
    )

    return parser.parse_args()


class DataVisualizer:
    """
    Advanced visualization tool for NIDS feature analysis.

    Provides comprehensive plotting capabilities for understanding
    feature distributions, attack patterns, and data quality.
    """

    def __init__(
        self, data_file: str, output_dir: str = "visualizations", style: str = "seaborn-v0_8"
    ):
        """
        Initialize the data visualizer.

        Args:
            data_file: Path to CSV file with extracted features
            output_dir: Directory to save visualization outputs
            style: Matplotlib style to use
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.df: pd.DataFrame | None = None

        plt.style.use(style)
        sns.set_palette("husl")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized DataVisualizer for {data_file}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset."""
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

            if self.df.empty:
                raise ValueError("Dataset is empty")

            if "Label" not in self.df.columns:
                logger.warning("No 'Label' column found - using last column as label")
                self.df.rename(columns={self.df.columns[-1]: "Label"}, inplace=True)

            return self.df

        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.data_file}") from e

    def plot_label_distribution(self, save: bool = True) -> Figure | None:
        """
        Plot the distribution of attack vs benign traffic.

        Args:
            save: Whether to save the plot

        Returns:
            Matplotlib figure object or None if no Label column
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        if "Label" not in self.df.columns:
            logger.warning("No 'Label' column found - cannot plot label distribution")
            return None

        label_col = self.df["Label"].copy()

        if set(label_col.unique()).issubset({0, 1}):
            label_col = label_col.map({0: "Benign", 1: "Malicious"})
            palette = {"Benign": "#2E8B57", "Malicious": "#DC143C"}
        else:
            palette = "husl"

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=label_col, palette=palette, edgecolor="black", linewidth=1, ax=ax)

        ax.set_xlabel("Traffic Type", fontsize=12, fontweight="bold")
        ax.set_ylabel("Number of Flows", fontsize=12, fontweight="bold")
        ax.set_title("Distribution of Traffic Types", fontsize=14, fontweight="bold")

        for container in ax.containers:
            ax.bar_label(container, fmt="{:,.0f}", fontweight="bold")  # type: ignore[arg-type]

        total = len(label_col)
        for patch in ax.patches:
            height = patch.get_height()  # type: ignore[attr-defined]
            percentage = (height / total) * 100
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,  # type: ignore[attr-defined]
                height / 2,
                f"{percentage:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
                fontsize=11,
            )

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "label_distribution.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Label distribution plot saved to {output_path}")

        return fig

    def plot_feature_histogram(
        self, feature: str, bins: int = 30, split_by_label: bool = True, save: bool = True
    ) -> Figure:
        """
        Plot histogram for a specific feature.

        Args:
            feature: Feature name to plot
            bins: Number of histogram bins
            split_by_label: Whether to split histogram by label
            save: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset")

        fig, ax = plt.subplots(figsize=(12, 6))

        if split_by_label and "Label" in self.df.columns:
            plot_data = self.df[[feature, "Label"]].copy()
            plot_data["Label_Name"] = plot_data["Label"].map(
                lambda x: "Benign" if x == 0 else "Malicious" if x == 1 else str(x)
            )

            sns.histplot(
                data=plot_data,
                x=feature,
                hue="Label_Name",
                bins=bins,
                stat="density",
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
            )
        else:
            sns.histplot(
                data=self.df,
                x=feature,
                bins=bins,
                stat="density",
                color="skyblue",
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
            )

        ax.set_xlabel(feature, fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title(f"Distribution of {feature}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{feature}_histogram.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Histogram for '{feature}' saved to {output_path}")

        return fig

    def plot_feature_scatter(
        self, feature_x: str, feature_y: str, sample_size: int | None = 5000, save: bool = True
    ) -> Figure:
        """
        Plot scatter plot for two features.

        Args:
            feature_x: X-axis feature
            feature_y: Y-axis feature
            sample_size: Number of points to sample (None for all)
            save: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        for feature in [feature_x, feature_y]:
            if feature not in self.df.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset")

        plot_df = self.df
        if sample_size and len(self.df) > sample_size:
            plot_df = self.df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} points for scatter plot")

        fig, ax = plt.subplots(figsize=(10, 8))

        if "Label" in plot_df.columns:
            plot_data = plot_df[[feature_x, feature_y, "Label"]].copy()
            plot_data["Label_Name"] = plot_data["Label"].map(
                lambda x: "Benign" if x == 0 else "Malicious" if x == 1 else str(x)
            )

            sns.scatterplot(
                data=plot_data,
                x=feature_x,
                y=feature_y,
                hue="Label_Name",
                palette={"Benign": "#2E8B57", "Malicious": "#DC143C"},
                alpha=0.6,
                s=20,
                edgecolor="black",
                linewidth=0.1,
                ax=ax,
            )
        else:
            sns.scatterplot(
                data=plot_df,
                x=feature_x,
                y=feature_y,
                color="skyblue",
                alpha=0.6,
                s=20,
                edgecolor="black",
                linewidth=0.1,
                ax=ax,
            )

        ax.set_xlabel(feature_x, fontsize=12, fontweight="bold")
        ax.set_ylabel(feature_y, fontsize=12, fontweight="bold")
        ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"{feature_x}_vs_{feature_y}_scatter.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Scatter plot saved to {output_path}")

        return fig

    def plot_correlation_matrix(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        feature_subset: list[str] | None = None,
        save: bool = True,
    ) -> Figure:
        """
        Plot correlation matrix for numeric features.

        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            feature_subset: Specific features to include (None for all numeric)
            save: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        if feature_subset:
            numeric_df = self.df[feature_subset]
        else:
            numeric_df = self.df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric features found for correlation analysis")

        corr_matrix = numeric_df.corr(method=method)
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(
            f"Feature Correlation Matrix ({method.title()})", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        if save:
            output_path = self.output_dir / f"correlation_matrix_{method}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Correlation matrix saved to {output_path}")

        return fig

    def plot_feature_importance_analysis(self, top_n: int = 20, save: bool = True) -> Figure:
        """
        Analyze and plot feature importance based on correlation with labels.

        Args:
            top_n: Number of top features to show
            save: Whether to save the plot

        Returns:
            Matplotlib figure object
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        if "Label" not in self.df.columns:
            raise ValueError("Label column required for feature importance analysis")

        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "Label" in numeric_features:
            numeric_features.remove("Label")

        correlations = {}
        for feature in numeric_features:
            corr = abs(self.df[feature].corr(self.df["Label"]))
            if not np.isnan(corr):
                correlations[feature] = corr

        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        importance_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.barplot(
            data=importance_df,
            y="Feature",
            x="Importance",
            color="skyblue",
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )

        ax.set_xlabel("Absolute Correlation with Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        for idx, row in importance_df.iterrows():
            ax.text(
                row["Importance"] + 0.01,
                int(idx),  # type: ignore[arg-type]
                f"{row['Importance']:.3f}",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "feature_importance.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {output_path}")

        return fig

    def generate_comprehensive_report(
        self, sample_features: list[str] | None = None
    ) -> dict[str, str | tuple[int, int] | list[str]]:
        """
        Generate a comprehensive visualization report.

        Args:
            sample_features: Specific features to analyze in detail

        Returns:
            Dictionary with report metadata
        """
        if self.df is None:
            self.load_data()

        assert self.df is not None, "Failed to load data"

        report = {
            "dataset_shape": self.df.shape,
            "output_directory": str(self.output_dir),
            "plots_generated": [],
        }

        logger.info("Generating comprehensive visualization report...")

        try:
            self.plot_label_distribution()
            report["plots_generated"].append("label_distribution.png")
        except Exception as e:
            logger.warning(f"Failed to generate label distribution plot: {e}")

        try:
            self.plot_feature_importance_analysis()
            report["plots_generated"].append("feature_importance.png")
        except Exception as e:
            logger.warning(f"Failed to generate feature importance plot: {e}")

        try:
            self.plot_correlation_matrix()
            report["plots_generated"].append("correlation_matrix_pearson.png")
        except Exception as e:
            logger.warning(f"Failed to generate correlation matrix: {e}")

        if sample_features is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "Label" in numeric_cols:
                numeric_cols.remove("Label")
            sample_features = numeric_cols[:5]

        for feature in sample_features:
            try:
                self.plot_feature_histogram(feature)
                report["plots_generated"].append(f"{feature}_histogram.png")
            except Exception as e:
                logger.warning(f"Failed to generate histogram for {feature}: {e}")

        logger.info(f"Report generated with {len(report['plots_generated'])} plots")
        logger.info(f"All visualizations saved to {self.output_dir}")

        return report


def main():
    """Command-line interface for data visualization."""

    args = parse_arguments()

    try:
        visualizer = DataVisualizer(args.data_file, args.output_dir)

        if args.report:
            report = visualizer.generate_comprehensive_report()
            logger.success("Comprehensive report generated")
            logger.info(f"Output directory: {report['output_directory']}")
            logger.info(f"Dataset shape: {report['dataset_shape']}")
            logger.info(f"Plots generated: {len(report['plots_generated'])}")

        else:
            plots_generated = 0

            if args.histogram:
                visualizer.plot_feature_histogram(args.histogram)
                plots_generated += 1

            if args.scatter:
                visualizer.plot_feature_scatter(args.scatter[0], args.scatter[1], args.sample_size)
                plots_generated += 1

            if args.correlation:
                visualizer.plot_correlation_matrix()
                plots_generated += 1

            if args.importance:
                visualizer.plot_feature_importance_analysis()
                plots_generated += 1

            if plots_generated == 0:
                logger.warning(
                    "No specific plots requested. Use --report for comprehensive analysis"
                )
                return 1

            logger.success(f"Generated {plots_generated} visualization(s)")
            logger.info(f"Saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
