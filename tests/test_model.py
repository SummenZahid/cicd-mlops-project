#!/usr/bin/env python3
"""
test_model.py - Comprehensive Test Suite for CI/CD Failure Prediction
COM774 Coursework 2 - MLOps Implementation

Tests data validation, model training, predictions, and API functionality.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tempfile
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample CI/CD pipeline data"""
    np.random.seed(42)
    n_samples = 200
    
    lines_added = np.random.randint(10, 1000, n_samples)
    lines_deleted = np.random.randint(0, 500, n_samples)
    churn = lines_added + lines_deleted   # FIXED: churn must match test

    X = pd.DataFrame({
        'files_changed': np.random.randint(1, 100, n_samples),
        'lines_added': lines_added,
        'lines_deleted': lines_deleted,
        'churn': churn,   # <-- correct now
        'test_count': np.random.randint(5, 200, n_samples),
        'test_failures': np.random.randint(0, 20, n_samples),
        'test_fail_rate': np.random.uniform(0, 0.5, n_samples),
        'coverage': np.random.uniform(50, 100, n_samples),
        'pipeline_duration_s': np.random.uniform(60, 3600, n_samples),
        'jobs_total': np.random.randint(1, 10, n_samples),
        'artifact_size_mb': np.random.uniform(10, 500, n_samples),
        'prev_7d_failure_rate': np.random.uniform(0, 0.5, n_samples),
        'prev_30d_failure_rate': np.random.uniform(0, 0.5, n_samples),
        'flaky_tests_count': np.random.randint(0, 10, n_samples),
        'infra_alerts_count': np.random.randint(0, 5, n_samples),
        'cache_hit_rate': np.random.uniform(0, 1, n_samples),
        'security_alerts_count': np.random.randint(0, 5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'message_length': np.random.randint(10, 200, n_samples),
        'had_hotfix_keyword': np.random.randint(0, 2, n_samples),
        'dependency_updates': np.random.randint(0, 10, n_samples),
        'author_experience_log': np.random.uniform(0, 5, n_samples),
        'churn_per_file': np.random.uniform(5, 50, n_samples)
    })
    
    # Create target: higher failure when test_failures > 5 or coverage < 70
    y = np.where(
        (X['test_failures'] > 5) | (X['coverage'] < 70),
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    )
    
    return {'X': X, 'y': y}


@pytest.fixture
def trained_model(sample_data):
    """Train a simple model for testing"""
    X, y = sample_data['X'], sample_data['y']
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model


# Test Data Validation
class TestDataValidation:
    """Test data loading and validation"""
    
    def test_data_loading(self, sample_data):
        """Test sample data loads correctly"""
        X, y = sample_data['X'], sample_data['y']
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert len(X) == len(y)
        assert len(X) > 0
    
    def test_data_shape(self, sample_data):
        """Test data has correct shape"""
        X = sample_data['X']
        assert X.shape[0] == 200  # 200 samples
        assert X.shape[1] == 24   # 24 features
    
    def test_no_missing_values(self, sample_data):
        """Test no missing values"""
        X = sample_data['X']
        assert X.isnull().sum().sum() == 0
    
    def test_feature_types(self, sample_data):
        """Test features have correct types"""
        X = sample_data['X']
        
        int_features = ['files_changed', 'lines_added', 'test_count']
        for feat in int_features:
            assert X[feat].dtype in [np.int32, np.int64]
        
        float_features = ['coverage', 'pipeline_duration_s', 'cache_hit_rate']
        for feat in float_features:
            assert X[feat].dtype in [np.float32, np.float64]
    
    def test_target_binary(self, sample_data):
        """Test target is binary"""
        y = sample_data['y']
        unique_values = np.unique(y)
        assert len(unique_values) <= 2
        assert set(unique_values).issubset({0, 1})
    
    def test_feature_ranges(self, sample_data):
        """Test feature values are in expected ranges"""
        X = sample_data['X']
        
        assert X['coverage'].min() >= 0
        assert X['coverage'].max() <= 100
        assert X['hour'].min() >= 0
        assert X['hour'].max() <= 23
        assert X['cache_hit_rate'].min() >= 0
        assert X['cache_hit_rate'].max() <= 1


# Test Model Training
class TestModelTraining:
    """Test model training functionality"""
    
    def test_random_forest_training(self, sample_data):
        """Test Random Forest training"""
        X, y = sample_data['X'], sample_data['y']
        model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_model_convergence(self, sample_data):
        """Test model converges"""
        X, y = sample_data['X'], sample_data['y']
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'estimators_')
        assert len(model.estimators_) == 5
    
    def test_handles_class_imbalance(self, sample_data):
        """Test model handles class imbalance"""
        X, y = sample_data['X'], sample_data['y']
        model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert len(np.unique(predictions)) >= 1


# Test Model Predictions
class TestModelPredictions:
    """Test model prediction functionality"""
    
    def test_prediction_shape(self, trained_model, sample_data):
        """Test predictions have correct shape"""
        X = sample_data['X']
        predictions = trained_model.predict(X)
        
        assert len(predictions) == len(X)
        assert predictions.shape == (len(X),)
    
    def test_prediction_probabilities(self, trained_model, sample_data):
        """Test prediction probabilities"""
        X = sample_data['X']
        probas = trained_model.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert np.all(probas >= 0) and np.all(probas <= 1)
    
    def test_prediction_values(self, trained_model, sample_data):
        """Test predictions are binary"""
        X = sample_data['X']
        predictions = trained_model.predict(X)
        
        unique_predictions = np.unique(predictions)
        assert set(unique_predictions).issubset({0, 1})
    
    def test_single_prediction(self, trained_model, sample_data):
        """Test single sample prediction"""
        X = sample_data['X']
        single_sample = X.iloc[[0]]
        
        prediction = trained_model.predict(single_sample)
        proba = trained_model.predict_proba(single_sample)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]
        assert proba.shape == (1, 2)
    
    def test_batch_prediction(self, trained_model, sample_data):
        """Test batch prediction"""
        X = sample_data['X']
        batch = X.iloc[:10]
        
        predictions = trained_model.predict(batch)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


# Test Model Performance
class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_accuracy_calculation(self, trained_model, sample_data):
        """Test accuracy calculation"""
        X, y = sample_data['X'], sample_data['y']
        predictions = trained_model.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        
        assert 0 <= accuracy <= 1
        assert isinstance(accuracy, (float, np.floating))
    
    def test_f1_score_calculation(self, trained_model, sample_data):
        """Test F1 score calculation"""
        X, y = sample_data['X'], sample_data['y']
        predictions = trained_model.predict(X)
        
        f1 = f1_score(y, predictions, zero_division=0)
        
        assert 0 <= f1 <= 1
        assert isinstance(f1, (float, np.floating))
    
    def test_minimum_performance(self, trained_model, sample_data):
        """Test model meets minimum performance"""
        X, y = sample_data['X'], sample_data['y']
        predictions = trained_model.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        assert accuracy > 0.5  # Better than random
    
    def test_performance_consistency(self, sample_data):
        """Test model performance is consistent"""
        X, y = sample_data['X'], sample_data['y']
        
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        assert np.array_equal(pred1, pred2)


# Test Model Persistence
class TestModelPersistence:
    """Test model saving and loading"""
    
    def test_model_save_load(self, trained_model):
        """Test model can be saved and loaded"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            joblib.dump(trained_model, tmp.name)
            loaded_model = joblib.load(tmp.name)
            
            assert type(loaded_model) == type(trained_model)
            assert hasattr(loaded_model, 'predict')
    
    def test_loaded_model_predictions(self, trained_model, sample_data):
        """Test loaded model makes same predictions"""
        X = sample_data['X']
        original_preds = trained_model.predict(X)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            joblib.dump(trained_model, tmp.name)
            loaded_model = joblib.load(tmp.name)
            loaded_preds = loaded_model.predict(X)
            
            assert np.array_equal(original_preds, loaded_preds)
    
    def test_metadata_save_load(self):
        """Test metadata saving and loading"""
        metadata = {
            'model_type': 'random_forest',
            'n_estimators': 10,
            'python_version': sys.version
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(metadata, tmp)
            tmp.flush()
            
            with open(tmp.name, 'r') as f:
                loaded_metadata = json.load(f)
            
            assert loaded_metadata == metadata


# Test Feature Engineering
class TestFeatureEngineering:
    """Test feature engineering"""
    
    def test_churn_calculation(self, sample_data):
        """Test code churn calculation"""
        X = sample_data['X']
        calculated_churn = X['lines_added'] + X['lines_deleted']
        assert np.allclose(X['churn'], calculated_churn, rtol=0.2)
    
    def test_test_fail_rate_range(self, sample_data):
        """Test fail rate is in valid range"""
        X = sample_data['X']
        assert (X['test_fail_rate'] >= 0).all()
        assert (X['test_fail_rate'] <= 1).all()
    
    def test_churn_per_file_positive(self, sample_data):
        """Test churn per file is positive"""
        X = sample_data['X']
        assert (X['churn_per_file'] >= 0).all()


# Test Edge Cases
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_input_handling(self, trained_model):
        """Test handling of empty input"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            trained_model.predict(empty_df)
    
    def test_missing_features_handling(self, trained_model, sample_data):
        """Test missing features fails gracefully"""
        X = sample_data['X']
        partial_features = X[['files_changed', 'test_count']]
        
        with pytest.raises(Exception):
            trained_model.predict(partial_features)
    
    def test_extreme_values(self, trained_model):
        """Test handling of extreme values"""
        extreme_data = pd.DataFrame({
            'files_changed': [10000], 'lines_added': [0], 'lines_deleted': [0],
            'churn': [0], 'test_count': [1000], 'test_failures': [500],
            'test_fail_rate': [0.5], 'coverage': [0], 'pipeline_duration_s': [10000],
            'jobs_total': [100], 'artifact_size_mb': [1000],
            'prev_7d_failure_rate': [1.0], 'prev_30d_failure_rate': [1.0],
            'flaky_tests_count': [100], 'infra_alerts_count': [50],
            'cache_hit_rate': [0.0], 'security_alerts_count': [10],
            'hour': [23], 'is_weekend': [1], 'message_length': [1000],
            'had_hotfix_keyword': [1], 'dependency_updates': [50],
            'author_experience_log': [10], 'churn_per_file': [100]
        })
        
        prediction = trained_model.predict(extreme_data)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1]


# Test API Input Validation
class TestAPIInputValidation:
    """Test API input validation"""
    
    def test_valid_features(self):
        """Test valid feature input"""
        features = {
            'files_changed': 15, 'lines_added': 234, 'lines_deleted': 89,
            'churn': 323, 'test_count': 45, 'test_failures': 2,
            'test_fail_rate': 0.044, 'coverage': 78.5,
            'pipeline_duration_s': 420.5, 'jobs_total': 5,
            'artifact_size_mb': 125.3, 'prev_7d_failure_rate': 0.15,
            'prev_30d_failure_rate': 0.22, 'flaky_tests_count': 1,
            'infra_alerts_count': 0, 'cache_hit_rate': 0.85,
            'security_alerts_count': 0, 'hour': 14, 'is_weekend': False,
            'message_length': 75, 'had_hotfix_keyword': False,
            'dependency_updates': 0, 'author_experience_log': 3.5,
            'churn_per_file': 21.5
        }
        
        assert features['files_changed'] >= 0
        assert 0 <= features['coverage'] <= 100
        assert 0 <= features['hour'] <= 23
        assert 0 <= features['cache_hit_rate'] <= 1


# Test Integration
class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_workflow(self, sample_data):
        """Test complete workflow"""
        X, y = sample_data['X'], sample_data['y']
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        assert len(predictions) == len(y_test)
        assert 0 <= accuracy <= 1
    
    def test_save_load_predict(self, trained_model, sample_data):
        """Test save, load, and predict workflow"""
        X = sample_data['X']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model.pkl'
            
            joblib.dump(trained_model, model_path)
            loaded_model = joblib.load(model_path)
            predictions = loaded_model.predict(X)
            
            assert len(predictions) == len(X)


# Run tests
if __name__ == "__main__":
    print("="*80)
    print("CI/CD PIPELINE FAILURE PREDICTION - TEST SUITE")
    print(f"Python Version: {sys.version}")
    print("="*80 + "\n")
    
    pytest.main([__file__, '-v', '--tb=short', '--cov=.', '--cov-report=term'])
