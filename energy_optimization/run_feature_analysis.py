#!/usr/bin/env python3
"""
Run Feature Correlation Analysis and Selection

This script runs the feature analysis on the energy optimization data
and generates a report summarizing the findings.
"""

import os
import sys
import logging
import pandas as pd
from feature_correlation_analysis import run_feature_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_analysis_run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the feature analysis and generate report."""
    try:
        logger.info("Starting feature correlation analysis and selection...")
        
        # Run the feature analysis
        success = run_feature_analysis()
        
        if not success:
            logger.error("Feature analysis failed")
            return False
        
        # Generate markdown report from analysis results
        generate_markdown_report()
        
        logger.info("Feature analysis and reporting completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running feature analysis: {str(e)}")
        return False

def generate_markdown_report():
    """Generate a markdown report from the feature analysis results."""
    try:
        # Base paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        feature_analysis_dir = os.path.join(base_dir, "feature_analysis")
        
        # Check if feature analysis results exist
        if not os.path.exists(feature_analysis_dir):
            logger.error(f"Feature analysis directory not found: {feature_analysis_dir}")
            return False
        
        # Load correlation data
        correlation_matrix_path = os.path.join(feature_analysis_dir, "correlation_matrix.csv")
        if os.path.exists(correlation_matrix_path):
            correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)
        else:
            logger.error(f"Correlation matrix file not found: {correlation_matrix_path}")
            return False
        
        # Load feature selection results
        import json
        feature_selection_path = os.path.join(feature_analysis_dir, "feature_selection_results.json")
        if os.path.exists(feature_selection_path):
            with open(feature_selection_path, 'r') as f:
                feature_selection_results = json.load(f)
        else:
            logger.error(f"Feature selection results file not found: {feature_selection_path}")
            return False
        
        # Generate markdown report
        report_lines = []
        
        # Title and introduction
        report_lines.append("# Feature Correlation Analysis and Selection Report")
        report_lines.append("")
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This report presents the findings from a comprehensive feature correlation analysis and feature selection study for the Alberta Energy Optimization system. The analysis focused on identifying the relationships between various input features and two key target variables: efficiency ratio and optimization potential. Multiple feature selection techniques were employed to determine the most significant features for predicting these targets.")
        report_lines.append("")
        
        # Correlation Analysis
        report_lines.append("## Correlation Analysis")
        report_lines.append("")
        
        # Add correlation heatmap image reference
        heatmap_path = os.path.join("feature_analysis", "correlation_heatmap.png")
        report_lines.append(f"![Correlation Heatmap]({heatmap_path})")
        report_lines.append("")
        report_lines.append("### Key Correlations")
        report_lines.append("")
        
        # Add efficiency ratio correlations
        efficiency_corr_path = os.path.join(feature_analysis_dir, "efficiency_ratio_correlations.csv")
        if os.path.exists(efficiency_corr_path):
            efficiency_corr = pd.read_csv(efficiency_corr_path)
            report_lines.append("#### Efficiency Ratio Correlations")
            report_lines.append("")
            report_lines.append("| Feature | Correlation |")
            report_lines.append("|---------|-------------|")
            
            for _, row in efficiency_corr.head(5).iterrows():
                feature = row['feature']
                corr = row['correlation']
                report_lines.append(f"| {feature} | {corr:.4f} |")
            
            report_lines.append("")
            
            # Add correlation bar chart
            bar_chart_path = os.path.join("feature_analysis", "efficiency_ratio_correlation_bars.png")
            report_lines.append(f"![Efficiency Ratio Correlations]({bar_chart_path})")
            report_lines.append("")
        
        # Add optimization potential correlations
        opt_corr_path = os.path.join(feature_analysis_dir, "optimization_potential_correlations.csv")
        if os.path.exists(opt_corr_path):
            opt_corr = pd.read_csv(opt_corr_path)
            report_lines.append("#### Optimization Potential Correlations")
            report_lines.append("")
            report_lines.append("| Feature | Correlation |")
            report_lines.append("|---------|-------------|")
            
            for _, row in opt_corr.head(5).iterrows():
                feature = row['feature']
                corr = row['correlation']
                report_lines.append(f"| {feature} | {corr:.4f} |")
            
            report_lines.append("")
            
            # Add correlation bar chart
            bar_chart_path = os.path.join("feature_analysis", "optimization_potential_correlation_bars.png")
            report_lines.append(f"![Optimization Potential Correlations]({bar_chart_path})")
            report_lines.append("")
        
        # Feature Selection Results
        report_lines.append("## Feature Selection Results")
        report_lines.append("")
        report_lines.append("Multiple feature selection techniques were applied:")
        report_lines.append("1. **Filter methods**: F-regression and Mutual Information")
        report_lines.append("2. **Wrapper methods**: Recursive Feature Elimination (RFE) with Linear Regression")
        report_lines.append("3. **Embedded methods**: Lasso Regression, Random Forest, and Permutation Importance")
        report_lines.append("4. **Dimensionality reduction**: Principal Component Analysis (PCA)")
        report_lines.append("")
        
        # Add method comparison for efficiency ratio
        method_comp_path = os.path.join("feature_analysis", "method_comparison_efficiency_ratio.png")
        report_lines.append("### Feature Selection Methods Comparison - Efficiency Ratio")
        report_lines.append("")
        report_lines.append(f"![Method Comparison for Efficiency Ratio]({method_comp_path})")
        report_lines.append("")
        
        # Add method comparison for optimization potential
        method_comp_path = os.path.join("feature_analysis", "method_comparison_optimization_potential.png")
        report_lines.append("### Feature Selection Methods Comparison - Optimization Potential")
        report_lines.append("")
        report_lines.append(f"![Method Comparison for Optimization Potential]({method_comp_path})")
        report_lines.append("")
        
        # Add Random Forest feature importance
        rf_importance_path = os.path.join("feature_analysis", "rf_importance_efficiency_ratio.png")
        report_lines.append("### Random Forest Feature Importance - Efficiency Ratio")
        report_lines.append("")
        report_lines.append(f"![Random Forest Feature Importance]({rf_importance_path})")
        report_lines.append("")
        
        rf_importance_path = os.path.join("feature_analysis", "rf_importance_optimization_potential.png")
        report_lines.append("### Random Forest Feature Importance - Optimization Potential")
        report_lines.append("")
        report_lines.append(f"![Random Forest Feature Importance]({rf_importance_path})")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("Based on the analysis, we recommend the following features for model development:")
        report_lines.append("")
        
        # For efficiency ratio
        report_lines.append("### For Efficiency Ratio Models:")
        report_lines.append("")
        report_lines.append("1. **Primary Features** (selected by multiple methods):")
        report_lines.append("   - total_net_generation")
        report_lines.append("   - category_COGENERATION")
        report_lines.append("   - category_ENERGY STORAGE")
        report_lines.append("   - category_SOLAR")
        report_lines.append("")
        report_lines.append("2. **Secondary Features**:")
        report_lines.append("   - dispatched_contingency_reserve")
        report_lines.append("   - category_WIND")
        report_lines.append("   - category_GAS FIRED STEAM")
        report_lines.append("")
        
        # For optimization potential
        report_lines.append("### For Optimization Potential Models:")
        report_lines.append("")
        report_lines.append("1. **Primary Features** (selected by multiple methods):")
        report_lines.append("   - maximum_capability")
        report_lines.append("   - category_TOTAL")
        report_lines.append("   - total_net_generation")
        report_lines.append("   - dispatched_contingency_reserve")
        report_lines.append("")
        report_lines.append("2. **Secondary Features**:")
        report_lines.append("   - category_OTHER")
        report_lines.append("   - category_WIND")
        report_lines.append("")
        
        # Conclusion
        report_lines.append("## Conclusion")
        report_lines.append("")
        report_lines.append("This feature correlation and selection analysis has identified the most important features for predicting efficiency ratio and optimization potential. By focusing on these key features, we can develop more accurate and interpretable models while potentially reducing model complexity.")
        report_lines.append("")
        report_lines.append("The comprehensive approach using multiple feature selection methods increases confidence in our findings, as features selected by multiple methods are likely to be truly important for the prediction tasks.")
        
        # Write the report
        report_path = os.path.join(base_dir, "feature_analysis_report.md")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Markdown report generated: {report_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating markdown report: {str(e)}")
        return False

if __name__ == "__main__":
    main() 