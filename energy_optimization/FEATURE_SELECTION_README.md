# Feature Selection for Energy Optimization

This directory contains the implementation of feature correlation analysis and selection for the Alberta Energy Optimization system. This README explains the purpose, components, and usage of the feature selection implementation.

## Overview

The feature selection implementation consists of the following components:

1. **Feature Correlation Analysis Script** (`feature_correlation_analysis.py`): Analyzes correlations between input features and target variables (efficiency ratio and optimization potential), and applies various feature selection techniques.

2. **Feature Analysis Report Generator** (`run_feature_analysis.py`): Runs the feature analysis and generates a comprehensive report with visualizations.

3. **Integration with Energy Optimizer** (`energy_optimizer.py`): The main energy optimization code has been updated to use the selected features for model training.

4. **Analysis Report** (`feature_analysis_report.md`): A detailed report of the feature analysis findings with recommendations.

## Key Findings

Our analysis revealed the following important features:

### For Efficiency Ratio Models:

1. **Primary Features**:
   - total_net_generation
   - category_COGENERATION
   - category_ENERGY STORAGE
   - category_SOLAR

2. **Secondary Features**:
   - dispatched_contingency_reserve
   - category_WIND
   - category_GAS FIRED STEAM

### For Optimization Potential Models:

1. **Primary Features**:
   - maximum_capability
   - category_TOTAL
   - total_net_generation
   - dispatched_contingency_reserve

2. **Secondary Features**:
   - category_OTHER
   - category_WIND

## Usage

### Running Feature Analysis

To run the feature analysis and generate a report:

```bash
python3 run_feature_analysis.py
```

This will:
1. Perform the feature correlation analysis
2. Apply multiple feature selection techniques
3. Generate visualizations for feature importance
4. Create a comprehensive report (`feature_analysis_report.md`)

### Using Feature Selection in Models

The feature selection has been integrated into the energy optimization pipeline. To use it:

```python
# In your code
from energy_optimizer import EnergyDataProcessor, EnergyOptimizationModel

# Initialize data processor
data_processor = EnergyDataProcessor(data_dir)
data_processor.load_data()

# Initialize optimization model with feature selection
model = EnergyOptimizationModel(data_processor)

# Train models with feature selection
model.train_models(use_feature_selection=True)
```

To disable feature selection and use all features:

```python
# Train models without feature selection
model.train_models(use_feature_selection=False)
```

## Feature Selection Methods

The implementation uses multiple feature selection techniques to identify the most important features:

1. **Filter Methods**:
   - F-regression: Identifies linear correlations between features and target variables
   - Mutual Information: Captures both linear and non-linear relationships

2. **Wrapper Methods**:
   - Recursive Feature Elimination (RFE): Recursively removes features to find the optimal feature subset

3. **Embedded Methods**:
   - Lasso Regression: Uses L1 regularization to identify important features
   - Random Forest Feature Importance: Uses tree-based importance measures
   - Permutation Importance: Measures feature importance by permuting feature values

4. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA): Reduces dimensionality while preserving variance

## Benefits

Using feature selection in the energy optimization pipeline provides several benefits:

1. **Improved Model Performance**: By focusing on the most relevant features, models can achieve better predictive performance.

2. **Reduced Model Complexity**: Fewer features lead to simpler models that are easier to interpret and less prone to overfitting.

3. **Enhanced Interpretability**: Understanding which features are most important provides insights into the factors that influence energy efficiency and optimization potential.

4. **Computational Efficiency**: Training models with fewer features requires less computational resources and time.

## Output Files

The feature analysis generates several output files in the `feature_analysis` directory:

- **Correlation Analysis**:
  - `correlation_matrix.csv`: Matrix of correlation coefficients between all features
  - `correlation_heatmap.png`: Visualization of the correlation matrix
  - `efficiency_ratio_correlations.csv` and `optimization_potential_correlations.csv`: Correlation values for each target
  - `efficiency_ratio_correlation_bars.png` and `optimization_potential_correlation_bars.png`: Bar charts of correlations

- **Feature Selection Results**:
  - `feature_selection_results.json`: Detailed results from all feature selection methods
  - Various visualization files for each method (e.g., `rf_importance_efficiency_ratio.png`, `method_comparison_efficiency_ratio.png`)

## Maintenance and Updates

To update the feature selection implementation:

1. **Add New Feature Selection Methods**: Extend the `FeatureAnalyzer` class in `feature_correlation_analysis.py` with new methods.

2. **Update Feature Sets**: Modify the feature lists in the `_apply_feature_selection` method in `energy_optimizer.py` if new analysis suggests different important features.

3. **Regenerate Analysis**: After making changes to the feature selection logic, run `run_feature_analysis.py` to regenerate the analysis and report. 