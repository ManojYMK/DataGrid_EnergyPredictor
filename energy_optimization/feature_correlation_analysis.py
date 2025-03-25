#!/usr/bin/env python3
"""
Feature Correlation Analysis and Selection for Energy Optimization

This script analyzes the correlation between input features and target variables
in the energy optimization system and implements various feature selection techniques.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import logging
from sklearn.inspection import permutation_importance
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """Class for analyzing feature correlations and performing feature selection."""
    
    def __init__(self, data_dir):
        """Initialize with the directory containing the CSV files."""
        self.data_dir = data_dir
        self.all_assets = None
        self.features = None
        self.targets = None
        self.feature_names = None
        self.results_dir = os.path.join(os.path.dirname(data_dir), "feature_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_data(self):
        """Load asset data from CSV file."""
        try:
            assets_path = os.path.join(self.data_dir, "all_assets.csv")
            if os.path.exists(assets_path):
                self.all_assets = pd.read_csv(assets_path)
                logger.info(f"Loaded asset data: {len(self.all_assets)} rows")
                return True
            else:
                logger.error(f"Asset data file not found: {assets_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_features_and_targets(self):
        """Prepare features and target variables for analysis."""
        try:
            if self.all_assets is None:
                logger.error("Data not loaded. Call load_data() first.")
                return False
            
            # One-hot encode the category
            category_dummies = pd.get_dummies(self.all_assets['category'], prefix='category')
            
            # Get numerical features
            numerical_features = self.all_assets[['maximum_capability', 'total_net_generation', 'dispatched_contingency_reserve']]
            
            # Combine with other features
            self.features = pd.concat([numerical_features, category_dummies], axis=1)
            self.feature_names = self.features.columns.tolist()
            
            # Calculate derived target variables
            self.all_assets['efficiency_ratio'] = np.where(
                self.all_assets['maximum_capability'] > 0,
                self.all_assets['total_net_generation'] / self.all_assets['maximum_capability'],
                0
            )
            
            self.all_assets['optimization_potential'] = self.all_assets['maximum_capability'] - self.all_assets['total_net_generation']
            
            # Set targets for analysis
            self.targets = {
                'efficiency_ratio': self.all_assets['efficiency_ratio'],
                'optimization_potential': self.all_assets['optimization_potential']
            }
            
            logger.info(f"Prepared features with shape: {self.features.shape}")
            logger.info(f"Feature names: {self.feature_names}")
            logger.info(f"Prepared targets: {list(self.targets.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preparing features and targets: {str(e)}")
            return False
    
    def analyze_correlations(self):
        """Analyze correlations between features and targets."""
        if self.features is None or self.targets is None:
            logger.error("Features and targets not prepared. Call prepare_features_and_targets() first.")
            return False
        
        try:
            # Create a combined dataframe for correlation analysis
            combined_df = self.features.copy()
            for target_name, target_values in self.targets.items():
                combined_df[target_name] = target_values
            
            # Calculate correlation matrix
            correlation_matrix = combined_df.corr()
            
            # Save correlation matrix to CSV
            correlation_matrix.to_csv(os.path.join(self.results_dir, "correlation_matrix.csv"))
            
            # Plot correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "correlation_heatmap.png"))
            plt.close()
            
            # For each target, show correlations sorted by absolute value
            for target_name in self.targets.keys():
                target_correlations = correlation_matrix[target_name].drop(target_name)
                abs_correlations = target_correlations.abs().sort_values(ascending=False)
                
                # Save to CSV
                pd.DataFrame({
                    'feature': abs_correlations.index,
                    'correlation': target_correlations[abs_correlations.index]
                }).to_csv(os.path.join(self.results_dir, f"{target_name}_correlations.csv"), index=False)
                
                # Plot bar chart of correlations
                plt.figure(figsize=(10, 8))
                target_correlations[abs_correlations.index].plot(kind='bar', color=[
                    'red' if x < 0 else 'green' for x in target_correlations[abs_correlations.index]
                ])
                plt.title(f'Feature Correlations with {target_name}')
                plt.xlabel('Features')
                plt.ylabel('Correlation Coefficient')
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{target_name}_correlation_bars.png"))
                plt.close()
                
                logger.info(f"Top 5 features correlated with {target_name}:")
                for feature, corr in zip(abs_correlations.index[:5], target_correlations[abs_correlations.index[:5]]):
                    logger.info(f"  {feature}: {corr:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return False
    
    def perform_feature_selection(self):
        """Perform various feature selection techniques."""
        if self.features is None or self.targets is None:
            logger.error("Features and targets not prepared. Call prepare_features_and_targets() first.")
            return False
        
        try:
            results = {}
            
            # Normalize features for analysis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.features)
            
            for target_name, target_values in self.targets.items():
                target_results = {}
                y = target_values.values
                
                # 1. Filter methods
                
                # 1.1 F-regression (linear correlation)
                f_selector = SelectKBest(f_regression, k=5)
                f_selector.fit(X_scaled, y)
                f_support = f_selector.get_support()
                f_selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if f_support[i]]
                f_scores = f_selector.scores_
                
                target_results['f_regression'] = {
                    'selected_features': f_selected_features,
                    'all_scores': {self.feature_names[i]: f_scores[i] for i in range(len(self.feature_names))}
                }
                
                # 1.2 Mutual Information
                mi_selector = SelectKBest(mutual_info_regression, k=5)
                mi_selector.fit(X_scaled, y)
                mi_support = mi_selector.get_support()
                mi_selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if mi_support[i]]
                mi_scores = mi_selector.scores_
                
                target_results['mutual_info'] = {
                    'selected_features': mi_selected_features,
                    'all_scores': {self.feature_names[i]: mi_scores[i] for i in range(len(self.feature_names))}
                }
                
                # 2. Wrapper methods
                
                # 2.1 Recursive Feature Elimination with Linear Regression
                rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
                rfe.fit(X_scaled, y)
                rfe_support = rfe.get_support()
                rfe_selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if rfe_support[i]]
                rfe_ranking = rfe.ranking_
                
                target_results['rfe'] = {
                    'selected_features': rfe_selected_features,
                    'feature_ranking': {self.feature_names[i]: rfe_ranking[i] for i in range(len(self.feature_names))}
                }
                
                # 3. Embedded methods
                
                # 3.1 Lasso Regression (L1 regularization)
                lasso = Lasso(alpha=0.01)
                lasso.fit(X_scaled, y)
                lasso_coefs = lasso.coef_
                
                # Get features with non-zero coefficients
                lasso_selected_features = [self.feature_names[i] for i in range(len(self.feature_names)) if abs(lasso_coefs[i]) > 0]
                
                target_results['lasso'] = {
                    'selected_features': lasso_selected_features,
                    'coefficients': {self.feature_names[i]: lasso_coefs[i] for i in range(len(self.feature_names))}
                }
                
                # 3.2 Random Forest Feature Importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_scaled, y)
                importances = rf.feature_importances_
                
                target_results['random_forest'] = {
                    'importances': {self.feature_names[i]: importances[i] for i in range(len(self.feature_names))},
                    'selected_features': [self.feature_names[i] for i in np.argsort(importances)[-5:]]
                }
                
                # 3.3 Permutation Importance
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_scaled, y)
                perm_importance = permutation_importance(rf_model, X_scaled, y, n_repeats=10, random_state=42)
                
                target_results['permutation'] = {
                    'importances': {self.feature_names[i]: perm_importance.importances_mean[i] for i in range(len(self.feature_names))},
                    'selected_features': [self.feature_names[i] for i in np.argsort(perm_importance.importances_mean)[-5:]]
                }
                
                # 4. Dimensionality Reduction
                
                # 4.1 PCA
                pca = PCA(n_components=0.95)  # Retain 95% of variance
                pca.fit(X_scaled)
                explained_variance = pca.explained_variance_ratio_
                
                target_results['pca'] = {
                    'n_components': pca.n_components_,
                    'explained_variance': explained_variance,
                    'cumulative_variance': np.cumsum(explained_variance)
                }
                
                # Store results for this target
                results[target_name] = target_results
            
            # Save results to JSON
            # Custom JSON encoder to handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return json.JSONEncoder.default(self, obj)
            
            # Convert results dictionary for JSON serialization
            # No need for manual conversion - we'll use the custom encoder
            
            with open(os.path.join(self.results_dir, "feature_selection_results.json"), 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            
            # Generate feature importance/selection plots
            self._plot_feature_selection_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing feature selection: {str(e)}")
            return False
    
    def _plot_feature_selection_results(self, results):
        """Generate plots for feature selection results."""
        try:
            for target_name, target_results in results.items():
                # 1. Plot F-regression scores
                f_scores = target_results['f_regression']['all_scores']
                self._plot_feature_scores(f_scores, f"F-Regression Scores for {target_name}", 
                                          f"f_regression_{target_name}.png")
                
                # 2. Plot Mutual Information scores
                mi_scores = target_results['mutual_info']['all_scores']
                self._plot_feature_scores(mi_scores, f"Mutual Information Scores for {target_name}", 
                                          f"mutual_info_{target_name}.png")
                
                # 3. Plot Lasso coefficients
                lasso_coefs = target_results['lasso']['coefficients']
                self._plot_feature_scores(lasso_coefs, f"Lasso Coefficients for {target_name}", 
                                          f"lasso_coefs_{target_name}.png", sort_by_abs=True)
                
                # 4. Plot Random Forest feature importances
                rf_importances = target_results['random_forest']['importances']
                self._plot_feature_scores(rf_importances, f"Random Forest Feature Importance for {target_name}", 
                                          f"rf_importance_{target_name}.png")
                
                # 5. Plot Permutation importance
                perm_importances = target_results['permutation']['importances']
                self._plot_feature_scores(perm_importances, f"Permutation Importance for {target_name}", 
                                          f"perm_importance_{target_name}.png")
                
                # 6. Plot PCA explained variance
                explained_variance = target_results['pca']['explained_variance']
                cumulative_variance = target_results['pca']['cumulative_variance']
                
                plt.figure(figsize=(10, 6))
                plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='blue')
                plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red', marker='o')
                plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.5)
                plt.title(f'PCA Explained Variance for {target_name}')
                plt.xlabel('Principal Components')
                plt.ylabel('Explained Variance Ratio')
                plt.legend(['Individual', 'Cumulative', '95% Variance'])
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"pca_variance_{target_name}.png"))
                plt.close()
                
                # 7. Compare feature selection methods
                self._plot_method_comparison(target_results, target_name)
                
        except Exception as e:
            logger.error(f"Error plotting feature selection results: {str(e)}")
    
    def _plot_feature_scores(self, scores_dict, title, filename, sort_by_abs=False):
        """Plot feature scores/importances."""
        try:
            # Convert to pandas Series
            scores = pd.Series(scores_dict)
            
            # Sort scores
            if sort_by_abs:
                scores = scores.reindex(scores.abs().sort_values(ascending=False).index)
            else:
                scores = scores.sort_values(ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 8))
            ax = scores.plot(kind='bar', color=['red' if x < 0 else 'blue' for x in scores])
            plt.title(title)
            plt.xlabel('Features')
            plt.ylabel('Score/Importance')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.xticks(rotation=90)
            
            # Add value labels
            for i, v in enumerate(scores):
                ax.text(i, v * (1.01 if v >= 0 else 0.99), f"{v:.3f}", ha='center', va='bottom' if v >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, filename))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature scores: {str(e)}")
    
    def _plot_method_comparison(self, target_results, target_name):
        """Compare selected features across different methods."""
        try:
            methods = ['f_regression', 'mutual_info', 'rfe', 'lasso', 'random_forest', 'permutation']
            method_names = ['F-Regression', 'Mutual Info', 'RFE', 'Lasso', 'Random Forest', 'Permutation']
            
            # Get all unique features selected by any method
            all_features = set()
            for method in methods:
                if method in target_results and 'selected_features' in target_results[method]:
                    all_features.update(target_results[method]['selected_features'])
            
            # Create a matrix of whether each feature was selected by each method
            feature_method_matrix = np.zeros((len(all_features), len(methods)))
            
            all_features = sorted(list(all_features))
            for i, feature in enumerate(all_features):
                for j, method in enumerate(methods):
                    if (method in target_results and 
                        'selected_features' in target_results[method] and 
                        feature in target_results[method]['selected_features']):
                        feature_method_matrix[i, j] = 1
            
            # Plot matrix
            plt.figure(figsize=(10, len(all_features) * 0.4 + 2))
            sns.heatmap(feature_method_matrix, annot=True, cmap='Blues', fmt='g',
                       xticklabels=method_names, yticklabels=all_features)
            plt.title(f'Feature Selection Methods Comparison for {target_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"method_comparison_{target_name}.png"))
            plt.close()
            
            # Calculate and plot feature selection frequency
            feature_freq = feature_method_matrix.sum(axis=1)
            feature_freq_df = pd.DataFrame({'Feature': all_features, 'Frequency': feature_freq})
            feature_freq_df = feature_freq_df.sort_values('Frequency', ascending=False)
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Frequency', y='Feature', data=feature_freq_df, color='skyblue')
            
            # Add annotations
            for i, row in enumerate(feature_freq_df.itertuples()):
                ax.text(row.Frequency + 0.1, i, str(row.Frequency), va='center')
                
            plt.title(f'Feature Selection Frequency for {target_name}')
            plt.xlabel('Number of Methods Selecting Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"feature_frequency_{target_name}.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting method comparison: {str(e)}")

def run_feature_analysis():
    """Run the feature correlation analysis and selection processes."""
    try:
        # Base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Data directory
        data_dir = os.path.join(base_dir, "data")
        
        # Create feature analyzer
        analyzer = FeatureAnalyzer(data_dir)
        
        # Load data
        if not analyzer.load_data():
            logger.error("Failed to load data")
            return False
        
        # Prepare features and targets
        if not analyzer.prepare_features_and_targets():
            logger.error("Failed to prepare features and targets")
            return False
        
        # Analyze correlations
        if not analyzer.analyze_correlations():
            logger.error("Failed to analyze correlations")
            return False
        
        # Perform feature selection
        results = analyzer.perform_feature_selection()
        if not results:
            logger.error("Failed to perform feature selection")
            return False
        
        logger.info("Feature analysis completed successfully")
        logger.info(f"Results saved to: {analyzer.results_dir}")
        
        # Report most important features for each target
        for target_name, target_results in results.items():
            logger.info(f"\nMost important features for {target_name}:")
            
            # Get features that were selected by at least 3 methods
            selected_by_methods = {}
            for method in ['f_regression', 'mutual_info', 'rfe', 'lasso', 'random_forest', 'permutation']:
                if method in target_results and 'selected_features' in target_results[method]:
                    for feature in target_results[method]['selected_features']:
                        if feature not in selected_by_methods:
                            selected_by_methods[feature] = []
                        selected_by_methods[feature].append(method)
            
            highly_selected = {k: v for k, v in selected_by_methods.items() if len(v) >= 3}
            
            if highly_selected:
                for feature, methods in sorted(highly_selected.items(), key=lambda x: len(x[1]), reverse=True):
                    logger.info(f"  {feature}: Selected by {len(methods)} methods ({', '.join(methods)})")
            else:
                logger.info("  No features were selected by multiple methods")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature analysis: {str(e)}")
        return False

if __name__ == "__main__":
    run_feature_analysis() 