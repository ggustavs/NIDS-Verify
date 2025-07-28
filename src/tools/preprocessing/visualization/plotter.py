"""
Data Visualization and Analysis Tools

Comprehensive visualization utilities for NIDS feature analysis,
including distribution plots, correlation analysis, and attack pattern visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Advanced visualization tool for NIDS feature analysis.
    
    Provides comprehensive plotting capabilities for understanding
    feature distributions, attack patterns, and data quality.
    """
    
    def __init__(self, data_file: str, output_dir: str = "visualizations", 
                 style: str = "seaborn-v0_8"):
        """
        Initialize the data visualizer.
        
        Args:
            data_file: Path to CSV file with extracted features
            output_dir: Directory to save visualization outputs
            style: Matplotlib style to use
        """
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.df: Optional[pd.DataFrame] = None
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataVisualizer for {data_file}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset."""
        try:
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            
            # Basic data validation
            if self.df.empty:
                raise ValueError("Dataset is empty")
            
            # Check for label column
            if "Label" not in self.df.columns:
                logger.warning("No 'Label' column found - some visualizations may be limited")
            
            return self.df
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.data_file}: {e}")
    
    def plot_label_distribution(self, save: bool = True) -> plt.Figure:
        """
        Plot the distribution of attack vs benign traffic.
        
        Args:
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if self.df is None:
            self.load_data()
        
        if "Label" not in self.df.columns:
            logger.warning("No 'Label' column found - cannot plot label distribution")
            return None
        
        # Count labels
        label_counts = self.df["Label"].value_counts().sort_index()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle different label formats
        if set(label_counts.index).issubset({0, 1}):
            # Binary labels (0/1)
            labels = ["Benign", "Malicious"]
            colors = ["#2E8B57", "#DC143C"]  # Sea green, crimson
        else:
            # String labels or other format
            labels = label_counts.index.tolist()
            colors = sns.color_palette("husl", len(labels))
        
        bars = ax.bar(range(len(label_counts)), label_counts.values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_xlabel("Traffic Type", fontsize=12, fontweight='bold')
        ax.set_ylabel("Number of Flows", fontsize=12, fontweight='bold')
        ax.set_title("Distribution of Traffic Types", fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar, count in zip(bars, label_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(label_counts.values) * 0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add percentage annotations
        total = label_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{percentage:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=11)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "label_distribution.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Label distribution plot saved to {output_path}")
        
        return fig
    
    def plot_feature_histogram(self, feature: str, bins: int = 30, 
                              split_by_label: bool = True, save: bool = True) -> plt.Figure:
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
        
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if split_by_label and "Label" in self.df.columns:
            # Split by label
            for label in sorted(self.df["Label"].unique()):
                subset = self.df[self.df["Label"] == label]
                label_name = "Benign" if label == 0 else "Malicious" if label == 1 else str(label)
                
                ax.hist(subset[feature].dropna(), bins=bins, alpha=0.7, 
                       label=label_name, density=True, edgecolor='black', linewidth=0.5)
        else:
            # Single histogram
            ax.hist(self.df[feature].dropna(), bins=bins, alpha=0.8, 
                   color='skyblue', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel("Density", fontsize=12, fontweight='bold')
        ax.set_title(f"Distribution of {feature}", fontsize=14, fontweight='bold')
        
        if split_by_label and "Label" in self.df.columns:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{feature}_histogram.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Histogram for '{feature}' saved to {output_path}")
        
        return fig
    
    def plot_feature_scatter(self, feature_x: str, feature_y: str, 
                           sample_size: Optional[int] = 5000, save: bool = True) -> plt.Figure:
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
        
        for feature in [feature_x, feature_y]:
            if feature not in self.df.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset")
        
        # Sample data if requested
        plot_df = self.df
        if sample_size and len(self.df) > sample_size:
            plot_df = self.df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} points for scatter plot")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if "Label" in plot_df.columns:
            # Color by label
            scatter = ax.scatter(plot_df[feature_x], plot_df[feature_y], 
                               c=plot_df["Label"], cmap='RdYlBu_r', 
                               alpha=0.6, s=20, edgecolors='black', linewidth=0.1)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Label", fontweight='bold')
        else:
            # Single color
            ax.scatter(plot_df[feature_x], plot_df[feature_y], 
                      alpha=0.6, s=20, color='skyblue', 
                      edgecolors='black', linewidth=0.1)
        
        ax.set_xlabel(feature_x, fontsize=12, fontweight='bold')
        ax.set_ylabel(feature_y, fontsize=12, fontweight='bold')
        ax.set_title(f"Scatter Plot: {feature_x} vs {feature_y}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{feature_x}_vs_{feature_y}_scatter.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scatter plot saved to {output_path}")
        
        return fig
    
    def plot_correlation_matrix(self, method: str = "pearson", 
                               feature_subset: Optional[List[str]] = None,
                               save: bool = True) -> plt.Figure:
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
        
        # Select numeric features
        if feature_subset:
            numeric_df = self.df[feature_subset]
        else:
            numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric features found for correlation analysis")
        
        # Compute correlation matrix
        corr_matrix = numeric_df.corr(method=method)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generate mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r", 
                   center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title(f"Feature Correlation Matrix ({method.title()})", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"correlation_matrix_{method}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {output_path}")
        
        return fig
    
    def plot_feature_importance_analysis(self, top_n: int = 20, save: bool = True) -> plt.Figure:
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
        
        if "Label" not in self.df.columns:
            raise ValueError("Label column required for feature importance analysis")
        
        # Get numeric features (exclude label)
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if "Label" in numeric_features:
            numeric_features.remove("Label")
        
        # Calculate correlation with labels
        correlations = {}
        for feature in numeric_features:
            corr = abs(self.df[feature].corr(self.df["Label"]))
            if not np.isnan(corr):
                correlations[feature] = corr
        
        # Sort by importance
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features, importances = zip(*top_features)
        
        bars = ax.barh(range(len(features)), importances, color='skyblue', 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Absolute Correlation with Label", fontsize=12, fontweight='bold')
        ax.set_title(f"Top {top_n} Most Important Features", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(importance + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "feature_importance.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        return fig
    
    def generate_comprehensive_report(self, sample_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive visualization report.
        
        Args:
            sample_features: Specific features to analyze in detail
            
        Returns:
            Dictionary with report metadata
        """
        if self.df is None:
            self.load_data()
        
        report = {
            "dataset_shape": self.df.shape,
            "output_directory": str(self.output_dir),
            "plots_generated": []
        }
        
        logger.info("Generating comprehensive visualization report...")
        
        # 1. Label distribution
        try:
            self.plot_label_distribution()
            report["plots_generated"].append("label_distribution.png")
        except Exception as e:
            logger.warning(f"Failed to generate label distribution plot: {e}")
        
        # 2. Feature importance
        try:
            self.plot_feature_importance_analysis()
            report["plots_generated"].append("feature_importance.png")
        except Exception as e:
            logger.warning(f"Failed to generate feature importance plot: {e}")
        
        # 3. Correlation matrix
        try:
            self.plot_correlation_matrix()
            report["plots_generated"].append("correlation_matrix_pearson.png")
        except Exception as e:
            logger.warning(f"Failed to generate correlation matrix: {e}")
        
        # 4. Sample feature histograms
        if sample_features is None:
            # Select a few interesting features automatically
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "Label" in numeric_cols:
                numeric_cols.remove("Label")
            sample_features = numeric_cols[:5]  # First 5 numeric features
        
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
        """
    )
    
    parser.add_argument("data_file", help="Path to the CSV file with extracted features")
    parser.add_argument("--output-dir", default="visualizations", 
                       help="Directory to save visualizations (default: visualizations)")
    parser.add_argument("--report", action="store_true", 
                       help="Generate comprehensive visualization report")
    parser.add_argument("--histogram", help="Generate histogram for specific feature")
    parser.add_argument("--scatter", nargs=2, metavar=("X", "Y"),
                       help="Generate scatter plot for two features")
    parser.add_argument("--correlation", action="store_true", 
                       help="Generate correlation matrix")
    parser.add_argument("--importance", action="store_true",
                       help="Generate feature importance analysis")
    parser.add_argument("--sample-size", type=int, default=5000,
                       help="Sample size for scatter plots (default: 5000)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        visualizer = DataVisualizer(args.data_file, args.output_dir)
        
        if args.report:
            # Generate comprehensive report
            report = visualizer.generate_comprehensive_report()
            print(f"✅ Comprehensive report generated!")
            print(f"📁 Output directory: {report['output_directory']}")
            print(f"📊 Dataset shape: {report['dataset_shape']}")
            print(f"🎨 Plots generated: {len(report['plots_generated'])}")
            
        else:
            # Generate specific plots
            plots_generated = 0
            
            if args.histogram:
                visualizer.plot_feature_histogram(args.histogram)
                plots_generated += 1
            
            if args.scatter:
                visualizer.plot_feature_scatter(args.scatter[0], args.scatter[1], 
                                              args.sample_size)
                plots_generated += 1
            
            if args.correlation:
                visualizer.plot_correlation_matrix()
                plots_generated += 1
            
            if args.importance:
                visualizer.plot_feature_importance_analysis()
                plots_generated += 1
            
            if plots_generated == 0:
                print("No specific plots requested. Use --report for comprehensive analysis.")
                return 1
            
            print(f"✅ Generated {plots_generated} visualization(s)")
            print(f"📁 Saved to: {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
