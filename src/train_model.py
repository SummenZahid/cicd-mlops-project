#!/usr/bin/env python3
"""
train_model.py - CI/CD Pipeline Failure Prediction Model Training
COM774 CW2 - MLOps Implementation
Author: Summen Zahid (B00996747)

This script trains multiple ML models to predict CI/CD pipeline failures.
It uses MLflow for experiment tracking and model versioning.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CICDFailurePredictor:
    """
    CI/CD Pipeline Failure Prediction Model Trainer
    
    Uses MLflow for experiment tracking and supports multiple ML algorithms.
    """
    
    def __init__(
        self,
        data_path: str = "training_data.csv",
        target_column: str = "success",
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the model trainer
        
        Args:
            data_path: Path to training data CSV
            target_column: Name of target column (success: 0=failure, 1=success)
            test_size: Test split ratio
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: list = []
        
        # Models and results
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Trainer initialized - Python {sys.version}")
    
    def load_data(self) -> None:
        """Load and validate the training dataset"""
        logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} samples with {len(self.df.columns)} columns")
        
        # Verify target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        # Check for missing values
        missing = self.df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values - filling with median")
            self.df = self.df.fillna(self.df.median(numeric_only=True))
        
        # Display target distribution
        logger.info(f"Target distribution:\n{self.df[self.target_column].value_counts()}")
        
        # Check class balance
        class_counts = self.df[self.target_column].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 2:
            logger.warning(f"Class imbalance detected: ratio {imbalance_ratio:.2f}")
    
    def prepare_features(self) -> None:
        """Split features and target, create train/test sets"""
        logger.info("Preparing features and target")
        
        # Separate features and target
        y = self.df[self.target_column]
        X = self.df.drop(columns=[self.target_column])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        logger.info(f"Using {len(self.feature_names)} features")
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        logger.info(f"Train distribution: {self.y_train.value_counts().to_dict()}")
    
    def train_random_forest(self) -> Dict[str, float]:
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Parameters
            params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1,
                'class_weight': 'balanced'  # Handle imbalance
            }
            mlflow.log_params(params)
            
            # Train
            start_time = datetime.now()
            model = RandomForestClassifier(**params)
            model.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['training_time_seconds'] = training_time
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Cross-validation
            cv_scores = self._cross_validate(model, "Random Forest")
            mlflow.log_metrics(cv_scores)
            
            # Feature importance
            self._log_feature_importance(model, "random_forest")
            
            # Confusion matrix
            self._log_confusion_matrix(self.y_test, y_pred, "random_forest")
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store
            self.models['random_forest'] = model
            self.results['random_forest'] = metrics
            
            logger.info(f"Random Forest - F1: {metrics['f1_score']:.4f}, "
                       f"AUC: {metrics['roc_auc_score']:.4f}")
            
            return metrics
    
    def train_xgboost(self) -> Dict[str, float]:
        """Train XGBoost classifier"""
        logger.info("Training XGBoost model...")
        
        with mlflow.start_run(run_name="xgboost"):
            # Calculate scale_pos_weight for imbalance
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'scale_pos_weight': scale_pos_weight
            }
            mlflow.log_params(params)
            
            # Train
            start_time = datetime.now()
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['training_time_seconds'] = training_time
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Cross-validation
            cv_scores = self._cross_validate(model, "XGBoost")
            mlflow.log_metrics(cv_scores)
            
            # Feature importance
            self._log_feature_importance(model, "xgboost")
            
            # Confusion matrix
            self._log_confusion_matrix(self.y_test, y_pred, "xgboost")
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store
            self.models['xgboost'] = model
            self.results['xgboost'] = metrics
            
            logger.info(f"XGBoost - F1: {metrics['f1_score']:.4f}, "
                       f"AUC: {metrics['roc_auc_score']:.4f}")
            
            return metrics
    
    def train_gradient_boosting(self) -> Dict[str, float]:
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting model...")
        
        with mlflow.start_run(run_name="gradient_boosting"):
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'subsample': 0.8,
                'random_state': self.random_state
            }
            mlflow.log_params(params)
            
            # Train
            start_time = datetime.now()
            model = GradientBoostingClassifier(**params)
            model.fit(self.X_train, self.y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
            metrics['training_time_seconds'] = training_time
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Cross-validation
            cv_scores = self._cross_validate(model, "Gradient Boosting")
            mlflow.log_metrics(cv_scores)
            
            # Feature importance
            self._log_feature_importance(model, "gradient_boosting")
            
            # Confusion matrix
            self._log_confusion_matrix(self.y_test, y_pred, "gradient_boosting")
            
            # Save model
            mlflow.sklearn.log_model(model, "model")
            
            # Store
            self.models['gradient_boosting'] = model
            self.results['gradient_boosting'] = metrics
            
            logger.info(f"Gradient Boosting - F1: {metrics['f1_score']:.4f}, "
                       f"AUC: {metrics['roc_auc_score']:.4f}")
            
            return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc_score': roc_auc_score(y_true, y_pred_proba)
        }
    
    def _cross_validate(self, model: Any, model_name: str) -> Dict[str, float]:
        """Perform stratified k-fold cross-validation"""
        logger.info(f"Running 5-fold cross-validation for {model_name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=cv, scoring='f1', n_jobs=-1
        )
        
        return {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'cv_f1_min': cv_scores.min(),
            'cv_f1_max': cv_scores.max()
        }
    
    def _log_feature_importance(self, model: Any, model_name: str) -> None:
        """Log feature importance plot"""
        if not hasattr(model, 'feature_importances_'):
            return
        
        # Get top 20 features
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 20 Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = f'/tmp/feature_importance_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(plot_path)
        logger.info(f"Logged feature importance for {model_name}")
    
    def _log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """Log confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Failure', 'Success'],
                    yticklabels=['Failure', 'Success'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = f'/tmp/confusion_matrix_{model_name}.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(plot_path)
        logger.info(f"Logged confusion matrix for {model_name}")
    
    def save_best_model(self, output_dir: str = "models") -> str:
        """Save the best performing model"""
        if not self.results:
            raise ValueError("No models trained yet")
        
        # Find best model by F1 score
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]
        best_metrics = self.results[best_model_name]
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / f"best_model_{best_model_name}.pkl"
        joblib.dump(best_model, model_file)
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'metrics': best_metrics,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'python_version': sys.version,
            'total_features': len(self.feature_names),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        import json
        metadata_file = output_path / "best_model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Best model ({best_model_name}) saved to {model_file}")
        logger.info(f"F1 Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {best_metrics['roc_auc_score']:.4f}")
        
        return str(model_file)
    
    def print_results_summary(self) -> None:
        """Print comparison of all trained models"""
        if not self.results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY - CI/CD PIPELINE FAILURE PREDICTION")
        print("="*80)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.round(4)
        
        print(comparison.to_string())
        print("="*80)
        
        # Best model
        best_model = max(self.results, key=lambda x: self.results[x]['f1_score'])
        print(f"\n🏆 BEST MODEL: {best_model.upper()}")
        print(f"   F1 Score:  {self.results[best_model]['f1_score']:.4f}")
        print(f"   ROC-AUC:   {self.results[best_model]['roc_auc_score']:.4f}")
        print(f"   Accuracy:  {self.results[best_model]['accuracy']:.4f}")
        print(f"   Precision: {self.results[best_model]['precision']:.4f}")
        print(f"   Recall:    {self.results[best_model]['recall']:.4f}")
        print("="*80 + "\n")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train CI/CD Pipeline Failure Prediction Models'
    )
    parser.add_argument(
        '--data',
        default='training_data.csv',
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--model',
        choices=['random_forest', 'xgboost', 'gradient_boosting', 'all'],
        default='all',
        help='Model to train'
    )
    parser.add_argument(
        '--mlflow-uri',
        default='./mlruns',
        help='MLflow tracking URI'
    )
    parser.add_argument(
        '--experiment',
        default='cicd_failure_prediction',
        help='MLflow experiment name'
    )
    
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    
    logger.info("="*80)
    logger.info("CI/CD PIPELINE FAILURE PREDICTION - MODEL TRAINING")
    logger.info(f"Python Version: {sys.version}")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = CICDFailurePredictor(data_path=args.data)
    
    # Load and prepare data
    trainer.load_data()
    trainer.prepare_features()
    
    # Train models
    if args.model in ['random_forest', 'all']:
        trainer.train_random_forest()
    
    if args.model in ['xgboost', 'all']:
        trainer.train_xgboost()
    
    if args.model in ['gradient_boosting', 'all']:
        trainer.train_gradient_boosting()
    
    # Print results
    trainer.print_results_summary()
    
    # Save best model
    model_path = trainer.save_best_model()
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"📊 Best model saved to: {model_path}")
    logger.info(f"📈 View results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()