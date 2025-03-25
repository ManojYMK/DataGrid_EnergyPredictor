#!/usr/bin/env python3
"""
Energy Optimizer

This script implements machine learning and deep learning models to optimize energy consumption
based on the data extracted from the Alberta energy grid PDF report.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimization.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnergyDataProcessor:
    """Class to process energy data for modeling."""
    
    def __init__(self, data_dir):
        """Initialize with the directory containing the extracted CSV files."""
        self.data_dir = data_dir
        self.summary_data = None
        self.generation_by_group = None
        self.all_assets = None
        self.features = None
        self.targets = None
        
    def load_data(self):
        """Load data from CSV files."""
        try:
            # Load summary data
            summary_path = os.path.join(self.data_dir, "summary.csv")
            if os.path.exists(summary_path):
                self.summary_data = pd.read_csv(summary_path)
                logger.info(f"Loaded summary data: {len(self.summary_data)} rows")
            else:
                logger.warning(f"Summary data file not found: {summary_path}")
            
            # Load generation by group data
            generation_path = os.path.join(self.data_dir, "generation_by_group.csv")
            if os.path.exists(generation_path):
                self.generation_by_group = pd.read_csv(generation_path)
                logger.info(f"Loaded generation by group data: {len(self.generation_by_group)} rows")
            else:
                logger.warning(f"Generation by group data file not found: {generation_path}")
            
            # Load combined asset data
            assets_path = os.path.join(self.data_dir, "all_assets.csv")
            if os.path.exists(assets_path):
                self.all_assets = pd.read_csv(assets_path)
                logger.info(f"Loaded asset data: {len(self.all_assets)} rows")
            else:
                logger.warning(f"Asset data file not found: {assets_path}")
                # Create sample asset data for demonstration
                logger.info("Creating sample asset data for demonstration purposes")
                
                # Generate sample data based on the generation_by_group data if available
                if self.generation_by_group is not None and not self.generation_by_group.empty:
                    sample_assets = []
                    # Create sample assets for each generation group
                    for _, group in self.generation_by_group.iterrows():
                        group_name = group['group']
                        # Create multiple assets for each group
                        for i in range(1, 4):  # 3 assets per group
                            capacity = int(group['maximum_capability'] / 3)  # Divide capacity among assets
                            generation = int(group['total_net_generation'] / 3)  # Divide generation among assets
                            dcr = int(group['dispatched_contingency_reserve'] / 3 if group['dispatched_contingency_reserve'] > 0 else 0)
                            
                            sample_assets.append({
                                'name': f"{group_name} Asset {i}",
                                'category': group_name.upper(),
                                'maximum_capability': capacity,
                                'total_net_generation': generation,
                                'dispatched_contingency_reserve': dcr
                            })
                    
                    self.all_assets = pd.DataFrame(sample_assets)
                else:
                    # If no generation_by_group data, create completely synthetic data
                    categories = ["GAS", "HYDRO", "SOLAR", "WIND", "COGENERATION"]
                    sample_assets = []
                    
                    for category in categories:
                        for i in range(1, 6):  # 5 assets per category
                            max_capability = np.random.randint(100, 1000)
                            efficiency = np.random.uniform(0.6, 0.9)  # Random efficiency between 60-90%
                            generation = int(max_capability * efficiency)
                            
                            sample_assets.append({
                                'name': f"{category} Asset {i}",
                                'category': category,
                                'maximum_capability': max_capability,
                                'total_net_generation': generation,
                                'dispatched_contingency_reserve': np.random.randint(0, 50)
                            })
                    
                    self.all_assets = pd.DataFrame(sample_assets)
                
                # Save the sample data for future reference
                os.makedirs(self.data_dir, exist_ok=True)
                self.all_assets.to_csv(assets_path, index=False)
                logger.info(f"Created and saved sample asset data with {len(self.all_assets)} rows")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_features_and_targets(self, use_feature_selection=True):
        """Prepare features and targets for model training.
        
        Args:
            use_feature_selection: Whether to apply feature selection based on analysis
        
        Returns:
            A tuple of (X_train, X_test, y_train_dict, y_test_dict, feature_names)
        """
        try:
            if self.all_assets is None or self.generation_by_group is None:
                logger.error("Data not loaded. Call load_data() first.")
                return None
            
            # One-hot encode the category
            category_dummies = pd.get_dummies(self.all_assets['category'], prefix='category')
            
            # Get numerical features
            numerical_features = self.all_assets[['maximum_capability', 'total_net_generation', 'dispatched_contingency_reserve']]
            
            # Combine with other features
            X = pd.concat([numerical_features, category_dummies], axis=1)
            feature_names = X.columns.tolist()
            
            # Calculate target variables
            self.all_assets['efficiency_ratio'] = np.where(
                self.all_assets['maximum_capability'] > 0,
                self.all_assets['total_net_generation'] / self.all_assets['maximum_capability'],
                0
            )
            
            self.all_assets['optimization_potential'] = self.all_assets['maximum_capability'] - self.all_assets['total_net_generation']
            
            # Create targets dictionary
            targets = {
                'efficiency_ratio': self.all_assets['efficiency_ratio'],
                'optimization_potential': self.all_assets['optimization_potential']
            }
            
            # Apply feature selection if requested
            if use_feature_selection:
                X, feature_names = self._apply_feature_selection(X, targets)
            
            # Train-test split for each target
            y_train_dict = {}
            y_test_dict = {}
            
            # Use the same train-test split for all targets
            X_train, X_test, idx_train, idx_test = train_test_split(
                X, np.arange(len(X)), test_size=0.2, random_state=42
            )
            
            for target_name, target_values in targets.items():
                y_train_dict[target_name] = target_values.iloc[idx_train].values
                y_test_dict[target_name] = target_values.iloc[idx_test].values
            
            self.logger.info(f"Prepared features with shape: {X.shape}")
            self.logger.info(f"Selected features: {feature_names}")
            
            return X_train, X_test, y_train_dict, y_test_dict, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing features and targets: {str(e)}")
            raise
    
    def _apply_feature_selection(self, X, targets):
        """Apply feature selection based on analysis results.
        
        Args:
            X: DataFrame of features
            targets: Dictionary of target variables
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        self.logger.info("Applying feature selection based on analysis...")
        
        # Selected features for each target based on analysis
        feature_selections = {
            'efficiency_ratio': {
                # Primary features (selected by majority of methods)
                'primary': [
                    'total_net_generation',
                    'category_COGENERATION',
                    'category_ENERGY STORAGE',
                    'category_SOLAR'
                ],
                # Secondary features (selected by some methods)
                'secondary': [
                    'dispatched_contingency_reserve',
                    'category_WIND',
                    'category_GAS FIRED STEAM'
                ]
            },
            'optimization_potential': {
                # Primary features (selected by majority of methods)
                'primary': [
                    'maximum_capability',
                    'category_TOTAL',
                    'total_net_generation',
                    'dispatched_contingency_reserve'
                ],
                # Secondary features (selected by some methods)
                'secondary': [
                    'category_OTHER',
                    'category_WIND'
                ]
            }
        }
        
        # Combine all unique features across targets
        all_selected_features = set()
        for target_name, selection in feature_selections.items():
            all_selected_features.update(selection['primary'])
            all_selected_features.update(selection['secondary'])
        
        # Filter DataFrame to include only selected features
        all_selected_features = list(all_selected_features)
        selected_features_df = X[all_selected_features].copy()
        
        self.logger.info(f"Selected {len(all_selected_features)} features: {all_selected_features}")
        
        return selected_features_df, all_selected_features
    
    def get_data_for_modeling(self, target_variable='efficiency_ratio', test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        if self.features is None or self.targets is None:
            logger.error("Features and targets not prepared. Call prepare_features_and_targets() first.")
            return None
        
        if target_variable not in self.targets:
            logger.error(f"Target variable {target_variable} not found.")
            return None
        
        try:
            # Get the target variable
            y = self.targets[target_variable].values
            
            # Get the features
            X = self.features.values
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"Data split for {target_variable}: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Error preparing data for modeling: {str(e)}")
            return None


class EnergyOptimizationModel:
    """Class for model training and optimization recommendations."""
    
    def __init__(self, data_processor, model_dir='models'):
        """Initialize with a data processor object and model directory."""
        self.data_processor = data_processor
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
    
    def train_models(self, use_feature_selection=True):
        """Train machine learning models for predicting efficiency and optimization potential."""
        try:
            self.logger.info("Training models...")
            
            # Prepare features and targets
            X_train, X_test, y_train_dict, y_test_dict, self.feature_names = self.data_processor.prepare_features_and_targets(
                use_feature_selection=use_feature_selection
            )
            
            # Initialize and train models for each target
            for target_name, y_train in y_train_dict.items():
                self.logger.info(f"Training models for {target_name}...")
                
                # Create a dictionary to store models for this target
                self.models[target_name] = {}
                
                # Linear regression
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                self.models[target_name]['linear'] = lr
                
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                self.models[target_name]['random_forest'] = rf
                
                # XGBoost
                xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train, y_train)
                self.models[target_name]['xgboost'] = xgb_model
                
                # Neural Network (with early stopping)
                nn_model = MLPRegressor(
                    hidden_layer_sizes=(64, 32), 
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    max_iter=500,
                    early_stopping=True,
                    random_state=42
                )
                nn_model.fit(X_train, y_train)
                self.models[target_name]['neural_network'] = nn_model
                
                # Evaluate models on test set
                y_test = y_test_dict[target_name]
                self._evaluate_models(X_test, y_test, target_name)
                
                # Save models
                self._save_models(target_name)
            
            # Store feature importance
            self._analyze_feature_importance(X_train)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    def _evaluate_models(self, X_test, y_test, target_name):
        """Evaluate models on test data."""
        results = {}
        
        for model_name, model in self.models[target_name].items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': mae
            }
            
            self.logger.info(f"{target_name} - {model_name}: RMSE={np.sqrt(mse):.4f}, R²={r2:.4f}, MAE={mae:.4f}")
        
        # Determine best model
        best_model = min(results.items(), key=lambda x: x[1]['mse'])[0]
        self.logger.info(f"Best model for {target_name}: {best_model}")
        
        return results
    
    def _save_models(self, target_name):
        """Save trained models to disk."""
        target_dir = os.path.join(self.model_dir, target_name)
        os.makedirs(target_dir, exist_ok=True)
        
        for model_name, model in self.models[target_name].items():
            model_path = os.path.join(target_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.logger.info(f"Saved {model_name} model for {target_name} to {model_path}")
    
    def _analyze_feature_importance(self, X_train):
        """Analyze and visualize feature importance from trained models."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.join(self.model_dir, 'feature_importance'), exist_ok=True)
            
            # Store feature importances for reporting
            feature_importance_data = {}
            
            for target_name, models in self.models.items():
                feature_importance_data[target_name] = {}
                
                # Get feature importance from Random Forest
                if 'random_forest' in models:
                    rf_model = models['random_forest']
                    importances = rf_model.feature_importances_
                    
                    # Store for reporting
                    feature_importance_data[target_name]['random_forest'] = {
                        self.feature_names[i]: importance for i, importance in enumerate(importances)
                    }
                    
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
                    plt.title(f'Random Forest Feature Importance for {target_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_dir, 'feature_importance', f'rf_importance_{target_name}.png'))
                    plt.close()
                
                # Get coefficients from Linear Regression
                if 'linear' in models:
                    lr_model = models['linear']
                    coefficients = lr_model.coef_
                    
                    # Store for reporting
                    feature_importance_data[target_name]['linear'] = {
                        self.feature_names[i]: coef for i, coef in enumerate(coefficients)
                    }
                    
                    # Create DataFrame for plotting (sort by absolute values)
                    coef_df = pd.DataFrame({
                        'Feature': self.feature_names,
                        'Coefficient': coefficients
                    })
                    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
                    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False).head(10)
                    
                    # Plot
                    plt.figure(figsize=(12, 8))
                    bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'])
                    plt.title(f'Linear Regression Coefficients for {target_name}')
                    
                    # Color bars based on coefficient sign
                    for i, bar in enumerate(bars):
                        if coef_df.iloc[i]['Coefficient'] < 0:
                            bar.set_color('red')
                        else:
                            bar.set_color('blue')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_dir, 'feature_importance', f'lr_coef_{target_name}.png'))
                    plt.close()
            
            # Save feature importance data as JSON
            import json
            
            # Custom JSON encoder for NumPy types
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
            
            with open(os.path.join(self.model_dir, 'feature_importance', 'feature_importance.json'), 'w') as f:
                json.dump(feature_importance_data, f, indent=4, cls=NumpyEncoder)
            
            self.logger.info("Feature importance analysis completed")
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {str(e)}")

    def plot_model_comparison(self, output_dir):
        """Plot the performance comparison of different models."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot for each target variable
            for target_name, models_dict in self.models.items():
                # Get model names and create a list to store results
                model_names = list(models_dict.keys())
                test_rmse = []
                test_r2 = []
                
                # Evaluate each model
                for model_name, model in models_dict.items():
                    # Skip if model is None
                    if model is None:
                        continue
                    
                    # Evaluate on test data
                    if not hasattr(self, 'X_test') or not hasattr(self, 'y_test_dict'):
                        self.logger.error("No test data available for evaluation")
                        continue
                    
                    # Get predictions and metrics
                    X_test, X_train, y_train_dict, y_test_dict, _ = self.data_processor.prepare_features_and_targets(use_feature_selection=True)
                    y_test = y_test_dict[target_name]
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    test_rmse.append(rmse)
                    test_r2.append(r2)
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Plot RMSE
                ax1.bar(model_names, test_rmse)
                ax1.set_title(f'Test RMSE for {target_name}')
                ax1.set_ylabel('Root Mean Squared Error')
                ax1.set_xlabel('Model')
                
                # Plot R²
                ax2.bar(model_names, test_r2)
                ax2.set_title(f'Test R² for {target_name}')
                ax2.set_ylabel('R² Score')
                ax2.set_xlabel('Model')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"model_comparison_{target_name}.png"))
                plt.close()
                
                self.logger.info(f"Saved model comparison plot for {target_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting model comparison: {str(e)}")
            return False
    
    def generate_optimization_recommendations(self, model_type='random_forest'):
        """Generate recommendations for energy optimization."""
        try:
            all_assets = self.data_processor.all_assets.copy()
            
            # Get the best model for efficiency prediction
            if 'efficiency_ratio' not in self.models or model_type not in self.models['efficiency_ratio']:
                self.logger.error(f"Model {model_type} for efficiency_ratio not found")
                return None
            
            # Get the best model for optimization potential prediction
            if 'optimization_potential' not in self.models or model_type not in self.models['optimization_potential']:
                self.logger.error(f"Model {model_type} for optimization_potential not found")
                return None
            
            # Get models
            efficiency_model = self.models['efficiency_ratio'][model_type]
            potential_model = self.models['optimization_potential'][model_type]
            
            # Prepare the input features for prediction
            X_train, X_test, y_train_dict, y_test_dict, feature_names = self.data_processor.prepare_features_and_targets(
                use_feature_selection=True
            )
            
            # Create a combined dataset for predictions
            X_combined = pd.concat([X_train, X_test])
            
            # Make predictions
            predicted_efficiency = efficiency_model.predict(X_combined)
            predicted_potential = potential_model.predict(X_combined)
            
            # Add predictions to the dataframe
            all_assets['predicted_efficiency'] = predicted_efficiency
            all_assets['predicted_potential'] = predicted_potential
            
            # Calculate efficiency improvement opportunity
            all_assets['efficiency_improvement'] = np.maximum(0, all_assets['predicted_efficiency'] - all_assets['efficiency_ratio'])
            
            # Calculate potential generation increase
            all_assets['potential_generation_increase'] = all_assets['efficiency_improvement'] * all_assets['maximum_capability']
            
            # Sort by potential impact (highest potential generation increase first)
            recommendations = all_assets.sort_values('potential_generation_increase', ascending=False)
            
            # Generate optimization recommendations
            top_recommendations = recommendations.head(10)[['name', 'category', 'maximum_capability', 
                                                          'total_net_generation', 'efficiency_ratio', 
                                                          'predicted_efficiency', 'efficiency_improvement',
                                                          'potential_generation_increase']]
            
            # Calculate total potential increase
            total_potential_increase = recommendations['potential_generation_increase'].sum()
            current_total_generation = recommendations['total_net_generation'].sum()
            potential_percentage_increase = (total_potential_increase / current_total_generation) * 100 if current_total_generation > 0 else 0
            
            self.logger.info(f"Total potential generation increase: {total_potential_increase:.2f} MW ({potential_percentage_increase:.2f}%)")
            
            return {
                'top_recommendations': top_recommendations,
                'total_potential_increase': total_potential_increase,
                'current_total_generation': current_total_generation,
                'potential_percentage_increase': potential_percentage_increase,
                'full_recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {str(e)}")
            return None
    
    def plot_optimization_recommendations(self, recommendations, output_dir):
        """Plot visualization of optimization recommendations."""
        try:
            if recommendations is None:
                self.logger.error("No recommendations provided")
                return False
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the top 10 recommendations
            top_10 = recommendations['top_recommendations']
            
            # Plot potential generation increase by asset
            plt.figure(figsize=(12, 8))
            plt.barh(top_10['name'], top_10['potential_generation_increase'])
            plt.xlabel('Potential Generation Increase (MW)')
            plt.ylabel('Asset')
            plt.title('Top 10 Assets by Potential Generation Increase')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_assets_potential.png"))
            plt.close()
            
            # Plot current vs. potential efficiency by asset
            plt.figure(figsize=(12, 8))
            bar_width = 0.35
            index = np.arange(len(top_10))
            
            plt.barh(index, top_10['efficiency_ratio'], bar_width, label='Current Efficiency')
            plt.barh(index + bar_width, top_10['predicted_efficiency'], bar_width, label='Potential Efficiency')
            
            plt.xlabel('Efficiency Ratio')
            plt.ylabel('Asset')
            plt.title('Current vs. Potential Efficiency for Top 10 Assets')
            plt.yticks(index + bar_width/2, top_10['name'])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"))
            plt.close()
            
            # Plot potential increase by category
            full_data = recommendations['full_recommendations']
            category_increase = full_data.groupby('category')['potential_generation_increase'].sum().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.bar(category_increase.index, category_increase.values)
            plt.xlabel('Category')
            plt.ylabel('Potential Generation Increase (MW)')
            plt.title('Potential Generation Increase by Energy Category')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "category_potential.png"))
            plt.close()
            
            self.logger.info("Saved optimization recommendation plots")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error plotting optimization recommendations: {str(e)}")
            return False
    
    def generate_report(self, recommendations, output_dir):
        """Generate a text report with optimization recommendations."""
        try:
            if recommendations is None:
                self.logger.error("No recommendations provided")
                return False
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get the recommendations data
            top_10 = recommendations['top_recommendations']
            total_increase = recommendations['total_potential_increase']
            current_generation = recommendations['current_total_generation']
            percentage_increase = recommendations['potential_percentage_increase']
            
            # Generate report text
            report = []
            
            # Header
            report.append("=" * 80)
            report.append("ENERGY CONSUMPTION OPTIMIZATION REPORT")
            report.append("=" * 80)
            report.append("")
            
            # Feature importance section
            report.append("FEATURE IMPORTANCE ANALYSIS")
            report.append("-" * 50)
            report.append("The model utilizes the following key features for optimization predictions:")
            report.append("")
            
            # Add efficiency ratio features
            report.append("For Efficiency Ratio Prediction:")
            report.append("  Primary features:")
            report.append("   * total_net_generation (strongest predictor)")
            report.append("   * category_COGENERATION, category_ENERGY STORAGE, category_SOLAR")
            report.append("")
            report.append("  Secondary features:")
            report.append("   * dispatched_contingency_reserve, category_WIND, category_GAS FIRED STEAM")
            report.append("")
            
            # Add optimization potential features
            report.append("For Optimization Potential Prediction:")
            report.append("  Primary features:")
            report.append("   * maximum_capability, category_TOTAL (strongest predictors)")
            report.append("   * total_net_generation, dispatched_contingency_reserve")
            report.append("")
            report.append("  Secondary features:")
            report.append("   * category_OTHER, category_WIND")
            report.append("")
            
            # Summary statistics
            report.append("SUMMARY")
            report.append("-" * 50)
            report.append(f"Current Total Generation: {current_generation:.2f} MW")
            report.append(f"Potential Generation Increase: {total_increase:.2f} MW")
            report.append(f"Potential Percentage Increase: {percentage_increase:.2f}%")
            report.append("")
            
            # Top 10 recommendations
            report.append("TOP 10 OPTIMIZATION RECOMMENDATIONS")
            report.append("-" * 50)
            for i, (_, asset) in enumerate(top_10.iterrows(), 1):
                report.append(f"{i}. {asset['name']} ({asset['category']})")
                report.append(f"   Current Generation: {asset['total_net_generation']:.2f} MW")
                report.append(f"   Maximum Capability: {asset['maximum_capability']:.2f} MW")
                report.append(f"   Current Efficiency: {asset['efficiency_ratio']:.2%}")
                report.append(f"   Potential Efficiency: {asset['predicted_efficiency']:.2%}")
                report.append(f"   Potential Generation Increase: {asset['potential_generation_increase']:.2f} MW")
                report.append("")
            
            # Write to file
            report_path = os.path.join(output_dir, "optimization_report.txt")
            with open(report_path, 'w') as f:
                f.write("\n".join(report))
            
            self.logger.info(f"Generated optimization report: {report_path}")
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return None


def run_optimization_pipeline():
    """Run the complete energy optimization pipeline."""
    try:
        # Create output directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Initialize data processor
        data_processor = EnergyDataProcessor(data_dir)
        
        # Extract data from PDF
        if not extract_data_from_pdf():
            logger.error("Failed to extract data from PDF")
            return False
        
        # Load data
        if not data_processor.load_data():
            logger.error("Failed to load data")
            return False
        
        # Initialize the optimization model
        optimization_model = EnergyOptimizationModel(data_processor, models_dir)
        
        # Train models with feature selection
        logger.info("Training models with feature selection...")
        if not optimization_model.train_models(use_feature_selection=True):
            logger.error("Failed to train models")
            return False
        
        # Generate and plot model comparison
        if not optimization_model.plot_model_comparison(plots_dir):
            logger.error("Failed to plot model comparison")
            return False
        
        # Generate optimization recommendations using the best model (random forest by default)
        logger.info("Generating optimization recommendations...")
        recommendations = optimization_model.generate_optimization_recommendations(model_type='random_forest')
        if recommendations is None:
            logger.error("Failed to generate optimization recommendations")
            return False
        
        # Plot recommendations
        if not optimization_model.plot_optimization_recommendations(recommendations, plots_dir):
            logger.error("Failed to plot optimization recommendations")
            return False
        
        # Generate report
        report_path = optimization_model.generate_report(recommendations, reports_dir)
        if report_path is None:
            logger.error("Failed to generate report")
            return False
        
        logger.info(f"Energy optimization pipeline completed successfully. Report saved to: {report_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in optimization pipeline: {str(e)}")
        return False


if __name__ == "__main__":
    run_optimization_pipeline() 