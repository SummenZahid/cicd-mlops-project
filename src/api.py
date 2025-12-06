#!/usr/bin/env python3
"""
api.py - FastAPI REST API for CI/CD Pipeline Failure Prediction
COM774 CW2 - MLOps Implementation
Author: Summen Zahid (B00996747)

Production-ready API for serving ML model predictions with monitoring.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model storage
MODEL = None
FEATURE_NAMES = None
MODEL_METADATA = None

# Prometheus metrics - create them lazily/cached to avoid duplicate registration
if PROMETHEUS_AVAILABLE:
    @lru_cache(maxsize=1)
    def get_prometheus_metrics():
        """
        Create and return prometheus metric objects only once per process.
        Using lru_cache prevents duplicate registration when the module is imported multiple times.
        """
        prediction_counter = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['predicted_outcome']
        )
        prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds'
        )
        api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method']
        )
        return {
            'prediction_counter': prediction_counter,
            'prediction_latency': prediction_latency,
            'api_requests': api_requests
        }

    _metrics = get_prometheus_metrics()
    prediction_counter = _metrics['prediction_counter']
    prediction_latency = _metrics['prediction_latency']
    api_requests = _metrics['api_requests']


# Pydantic models for request/response validation
class PipelineFeatures(BaseModel):
    """
    Input features for CI/CD pipeline failure prediction
    Based on your actual data structure
    """
    files_changed: int = Field(..., ge=0, description="Number of files changed")
    lines_added: int = Field(..., ge=0, description="Lines of code added")
    lines_deleted: int = Field(..., ge=0, description="Lines of code deleted")
    churn: int = Field(..., ge=0, description="Code churn (added + deleted)")
    test_count: int = Field(..., ge=0, description="Number of tests")
    test_failures: int = Field(0, ge=0, description="Number of failed tests")
    test_fail_rate: float = Field(0.0, ge=0.0, le=1.0, description="Test failure rate")
    coverage: float = Field(..., ge=0.0, le=100.0, description="Code coverage %")
    pipeline_duration_s: float = Field(..., gt=0, description="Pipeline duration in seconds")
    jobs_total: int = Field(..., ge=1, description="Total number of jobs")
    artifact_size_mb: float = Field(..., ge=0, description="Build artifact size in MB")
    prev_7d_failure_rate: float = Field(0.0, ge=0.0, le=1.0, description="7-day failure rate")
    prev_30d_failure_rate: float = Field(0.0, ge=0.0, le=1.0, description="30-day failure rate")
    flaky_tests_count: int = Field(0, ge=0, description="Number of flaky tests")
    infra_alerts_count: int = Field(0, ge=0, description="Infrastructure alerts")
    cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0, description="Cache hit rate")
    security_alerts_count: int = Field(0, ge=0, description="Security alerts")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    is_weekend: bool = Field(False, description="Is weekend?")
    message_length: int = Field(..., ge=0, description="Commit message length")
    had_hotfix_keyword: bool = Field(False, description="Has hotfix keyword")
    dependency_updates: int = Field(0, ge=0, description="Number of dependency updates")
    author_experience_log: float = Field(0.0, ge=0.0, description="Log of author experience")
    churn_per_file: float = Field(0.0, ge=0.0, description="Code churn per file")

    class Config:
        json_schema_extra = {
            "example": {
                "files_changed": 15,
                "lines_added": 234,
                "lines_deleted": 89,
                "churn": 323,
                "test_count": 45,
                "test_failures": 2,
                "test_fail_rate": 0.044,
                "coverage": 78.5,
                "pipeline_duration_s": 420.5,
                "jobs_total": 5,
                "artifact_size_mb": 125.3,
                "prev_7d_failure_rate": 0.15,
                "prev_30d_failure_rate": 0.22,
                "flaky_tests_count": 1,
                "infra_alerts_count": 0,
                "cache_hit_rate": 0.85,
                "security_alerts_count": 0,
                "hour": 14,
                "is_weekend": False,
                "message_length": 75,
                "had_hotfix_keyword": False,
                "dependency_updates": 0,
                "author_experience_log": 3.5,
                "churn_per_file": 21.5
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model"""
    success_probability: float = Field(..., description="Probability of success (0-1)")
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    predicted_outcome: str = Field(..., description="Predicted outcome (success/failure)")
    risk_level: str = Field(..., description="Risk level (HIGH/MODERATE/LOW)")
    confidence: float = Field(..., description="Prediction confidence")
    recommendation: str = Field(..., description="Actionable recommendation")
    timestamp: str = Field(..., description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success_probability": 0.85,
                "failure_probability": 0.15,
                "predicted_outcome": "success",
                "risk_level": "LOW",
                "confidence": 0.85,
                "recommendation": "✅ Low risk. Safe to proceed with deployment.",
                "timestamp": "2025-12-06T19:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    pipelines: List[PipelineFeatures] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_count: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    average_success_probability: float


class ModelInfo(BaseModel):
    """Model information"""
    model_type: str
    version: str
    python_version: str
    features_count: int
    metrics: Dict[str, float]
    last_updated: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    python_version: str


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("="*80)
    logger.info("CI/CD FAILURE PREDICTION API - STARTING")
    logger.info(f"Python Version: {sys.version}")
    logger.info("="*80)

    load_model()

    yield

    # Shutdown
    logger.info("Shutting down API")


def load_model(model_dir: str = "models") -> None:
    """Load the trained model and metadata"""
    global MODEL, FEATURE_NAMES, MODEL_METADATA

    model_path = Path(model_dir)

    # Find model file
    model_files = list(model_path.glob("best_model_*.pkl"))
    if not model_files:
        logger.error(f"No model found in {model_dir}")
        raise FileNotFoundError(f"No model file found in {model_dir}")

    model_file = model_files[0]
    logger.info(f"Loading model from {model_file}")

    try:
        MODEL = joblib.load(model_file)
        logger.info(f"Model loaded successfully: {type(MODEL).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load metadata
    metadata_file = model_path / "best_model_metadata.json"
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            MODEL_METADATA = json.load(f)
        FEATURE_NAMES = MODEL_METADATA.get('feature_names', [])
        logger.info(f"Loaded metadata: {len(FEATURE_NAMES)} features")
    else:
        logger.warning("No metadata file found")
        FEATURE_NAMES = []


def prepare_features(features: PipelineFeatures) -> pd.DataFrame:
    """
    Prepare features for prediction
    Ensures feature alignment with training data
    """
    # Convert to dict
    feature_dict = features.model_dump()

    # Convert boolean to int
    feature_dict['is_weekend'] = int(feature_dict['is_weekend'])
    feature_dict['had_hotfix_keyword'] = int(feature_dict['had_hotfix_keyword'])

    # Create DataFrame
    df = pd.DataFrame([feature_dict])

    # If we have feature names from training, align features
    if FEATURE_NAMES:
        # Add missing features with zeros (one-hot encoded features)
        missing_features = set(FEATURE_NAMES) - set(df.columns)
        for feat in missing_features:
            df[feat] = 0

        # Reorder columns to match training
        df = df[FEATURE_NAMES]

    return df


def get_risk_level(success_prob: float) -> str:
    """Determine risk level based on success probability"""
    if success_prob >= 0.7:
        return "LOW"
    elif success_prob >= 0.4:
        return "MODERATE"
    else:
        return "HIGH"


def get_recommendation(risk_level: str, success_prob: float) -> str:
    """Get actionable recommendation based on risk"""
    if risk_level == "LOW":
        return "✅ Low risk. Safe to proceed with deployment."
    elif risk_level == "MODERATE":
        return "⚠️ Moderate risk. Review code changes and test results before proceeding."
    else:
        return "🚨 High risk of failure. Recommend thorough review and additional testing."


# Initialize FastAPI app
app = FastAPI(
    title="CI/CD Pipeline Failure Prediction API",
    description="Machine Learning API for predicting CI/CD pipeline failures",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint"""
    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="/", method="GET").inc()

    return {
        "message": "CI/CD Pipeline Failure Prediction API",
        "version": "1.0.0",
        "author": "Summen Zahid (B00996747)",
        "course": "COM774 - Intelligence Engineering and Infrastructure",
        "python_version": sys.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="/health", method="GET").inc()

    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        timestamp=datetime.now().isoformat(),
        python_version=sys.version
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(features: PipelineFeatures) -> PredictionResponse:
    """
    Predict CI/CD pipeline outcome
    """
    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="/predict", method="POST").inc()

    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        # Prepare features
        X = prepare_features(features)

        # Make prediction
        if PROMETHEUS_AVAILABLE:
            with prediction_latency.time():
                prediction = MODEL.predict(X)[0]
                probabilities = MODEL.predict_proba(X)[0]
        else:
            prediction = MODEL.predict(X)[0]
            probabilities = MODEL.predict_proba(X)[0]

        # Get probabilities (0=failure, 1=success)
        failure_prob = float(probabilities[0])
        success_prob = float(probabilities[1])

        # Determine outcome and risk
        predicted_outcome = "success" if prediction == 1 else "failure"
        risk_level = get_risk_level(success_prob)
        confidence = max(success_prob, failure_prob)
        recommendation = get_recommendation(risk_level, success_prob)

        # Update metrics
        if PROMETHEUS_AVAILABLE:
            prediction_counter.labels(predicted_outcome=predicted_outcome).inc()

        # Create response
        response = PredictionResponse(
            success_probability=success_prob,
            failure_probability=failure_prob,
            predicted_outcome=predicted_outcome,
            risk_level=risk_level,
            confidence=confidence,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Prediction: {predicted_outcome}, Risk: {risk_level}, "
                    f"Success prob: {success_prob:.3f}")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction for multiple pipelines
    """
    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="/batch_predict", method="POST").inc()

    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        predictions = []
        risk_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
        total_success_prob = 0.0

        for pipeline_features in request.pipelines:
            # Make individual prediction
            pred_response = await predict(pipeline_features)
            predictions.append(pred_response)
            risk_counts[pred_response.risk_level] += 1
            total_success_prob += pred_response.success_probability

        avg_success_prob = total_success_prob / len(predictions)

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            high_risk_count=risk_counts["HIGH"],
            moderate_risk_count=risk_counts["MODERATE"],
            low_risk_count=risk_counts["LOW"],
            average_success_probability=avg_success_prob
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Get model information and metadata"""
    if PROMETHEUS_AVAILABLE:
        api_requests.labels(endpoint="/model/info", method="GET").inc()

    if MODEL is None or MODEL_METADATA is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return ModelInfo(
        model_type=MODEL_METADATA.get('model_type', 'unknown'),
        version="1.0.0",
        python_version=sys.version,
        features_count=len(FEATURE_NAMES),
        metrics=MODEL_METADATA.get('metrics', {}),
        last_updated=MODEL_METADATA.get('training_date', 'unknown')
    )


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


def main():
    """Run the API server"""
    logger.info("Starting FastAPI server...")
    logger.info(f"Python version: {sys.version}")
    logger.info("Prometheus metrics: " + ("enabled" if PROMETHEUS_AVAILABLE else "disabled"))

    # When running the script directly (python src/api.py), pass the app object
    # to uvicorn.run to avoid uvicorn importing the module by name again
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
