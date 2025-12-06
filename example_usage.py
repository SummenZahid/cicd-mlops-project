#!/usr/bin/env python3
"""
example_usage.py - Example usage of CI/CD Failure Prediction Model
COM774 CW2 - Quick testing and demonstration
"""

import pandas as pd
import joblib
import json

# Example 1: Load and use the model directly
def example_direct_prediction():
    """Example of direct model usage"""
    print("="*80)
    print("EXAMPLE 1: Direct Model Prediction")
    print("="*80)
    
    # Load model
    model = joblib.load('models/best_model_random_forest.pkl')
    
    # Load metadata to get feature names
    with open('models/best_model_metadata.json') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    
    # Create example pipeline data
    example_data = {
        'files_changed': 15,
        'lines_added': 234,
        'lines_deleted': 89,
        'churn': 323,
        'test_count': 45,
        'test_failures': 2,
        'test_fail_rate': 0.044,
        'coverage': 78.5,
        'pipeline_duration_s': 420.5,
        'jobs_total': 5,
        'artifact_size_mb': 125.3,
        'prev_7d_failure_rate': 0.15,
        'prev_30d_failure_rate': 0.22,
        'flaky_tests_count': 1,
        'infra_alerts_count': 0,
        'cache_hit_rate': 0.85,
        'security_alerts_count': 0,
        'hour': 14,
        'is_weekend': 0,
        'message_length': 75,
        'had_hotfix_keyword': 0,
        'dependency_updates': 0,
        'author_experience_log': 3.5,
        'churn_per_file': 21.5
    }
    
    # Create DataFrame
    df = pd.DataFrame([example_data])
    
    # Add missing one-hot encoded features with zeros
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure column order matches training
    df = df[feature_names]
    
    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    # Display results
    print(f"\n📊 Prediction Results:")
    print(f"   Predicted Outcome: {'✅ SUCCESS' if prediction == 1 else '❌ FAILURE'}")
    print(f"   Success Probability: {probabilities[1]:.1%}")
    print(f"   Failure Probability: {probabilities[0]:.1%}")
    
    # Risk assessment
    if probabilities[1] >= 0.7:
        risk = "LOW 🟢"
        recommendation = "Safe to proceed"
    elif probabilities[1] >= 0.4:
        risk = "MODERATE 🟡"
        recommendation = "Review before proceeding"
    else:
        risk = "HIGH 🔴"
        recommendation = "Thorough review required"
    
    print(f"   Risk Level: {risk}")
    print(f"   Recommendation: {recommendation}\n")


# Example 2: Batch predictions
def example_batch_prediction():
    """Example of batch predictions"""
    print("="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    # Load model
    model = joblib.load('models/best_model_random_forest.pkl')
    
    # Load metadata
    with open('models/best_model_metadata.json') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    
    # Create multiple test cases
    test_cases = [
        {
            'name': 'Low Risk Pipeline',
            'data': {
                'files_changed': 5, 'lines_added': 50, 'lines_deleted': 20,
                'churn': 70, 'test_count': 30, 'test_failures': 0,
                'test_fail_rate': 0.0, 'coverage': 95.0,
                'pipeline_duration_s': 200, 'jobs_total': 3,
                'artifact_size_mb': 30, 'prev_7d_failure_rate': 0.05,
                'prev_30d_failure_rate': 0.08
            }
        },
        {
            'name': 'High Risk Pipeline',
            'data': {
                'files_changed': 50, 'lines_added': 1000, 'lines_deleted': 500,
                'churn': 1500, 'test_count': 100, 'test_failures': 15,
                'test_fail_rate': 0.15, 'coverage': 45.0,
                'pipeline_duration_s': 1800, 'jobs_total': 8,
                'artifact_size_mb': 200, 'prev_7d_failure_rate': 0.35,
                'prev_30d_failure_rate': 0.40
            }
        },
        {
            'name': 'Moderate Risk Pipeline',
            'data': {
                'files_changed': 20, 'lines_added': 300, 'lines_deleted': 100,
                'churn': 400, 'test_count': 50, 'test_failures': 3,
                'test_fail_rate': 0.06, 'coverage': 70.0,
                'pipeline_duration_s': 600, 'jobs_total': 5,
                'artifact_size_mb': 80, 'prev_7d_failure_rate': 0.20,
                'prev_30d_failure_rate': 0.25
            }
        }
    ]
    
    print(f"\n📊 Testing {len(test_cases)} pipelines:\n")
    
    for case in test_cases:
        # Add default values for missing fields
        data = case['data']
        defaults = {
            'flaky_tests_count': 0, 'infra_alerts_count': 0,
            'cache_hit_rate': 0.8, 'security_alerts_count': 0,
            'hour': 12, 'is_weekend': 0, 'message_length': 50,
            'had_hotfix_keyword': 0, 'dependency_updates': 0,
            'author_experience_log': 3.0, 'churn_per_file': 15.0
        }
        data.update({k: v for k, v in defaults.items() if k not in data})
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Add missing features
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        
        # Predict
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]
        
        # Display
        outcome = "✅ SUCCESS" if prediction == 1 else "❌ FAILURE"
        print(f"   {case['name']}")
        print(f"      → Prediction: {outcome} ({proba:.1%} success)")
        print()


# Example 3: Feature importance
def example_feature_importance():
    """Show top features"""
    print("="*80)
    print("EXAMPLE 3: Feature Importance Analysis")
    print("="*80)
    
    # Load model
    model = joblib.load('models/best_model_random_forest.pkl')
    
    # Load metadata
    with open('models/best_model_metadata.json') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔝 Top 15 Most Important Features:\n")
    for i, row in importance.head(15).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"   {row['feature']:30s} {bar} {row['importance']:.4f}")
    
    print()


# Example 4: Model information
def example_model_info():
    """Display model information"""
    print("="*80)
    print("EXAMPLE 4: Model Information")
    print("="*80)
    
    # Load metadata
    with open('models/best_model_metadata.json') as f:
        metadata = json.load(f)
    
    print(f"\n📈 Model Details:")
    print(f"   Model Type: {metadata['model_type'].upper()}")
    print(f"   Features: {metadata['total_features']}")
    print(f"   Training Samples: {metadata['train_samples']}")
    print(f"   Test Samples: {metadata['test_samples']}")
    print(f"   Training Date: {metadata['training_date']}")
    
    print(f"\n📊 Performance Metrics:")
    for metric, value in metadata['metrics'].items():
        if not metric.endswith('_seconds'):
            print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "🚀 CI/CD PIPELINE FAILURE PREDICTION - EXAMPLES\n")
    
    try:
        example_direct_prediction()
        example_batch_prediction()
        example_feature_importance()
        example_model_info()
        
        print("="*80)
        print("✅ All examples completed successfully!")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've trained the model first:")
        print("   python3 src/train_model.py --data data/training_data.csv --model all\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")