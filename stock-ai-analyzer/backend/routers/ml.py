"""
ML Prediction API routes.
"""

from fastapi import APIRouter, HTTPException, Query
import logging

from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["ML Predictions"])


@router.get("/{symbol}")
async def get_prediction(symbol: str):
    """
    Get ML-based directional prediction for a stock.
    Shows probability of up/down move over 1 day, 1 week, 1 month.
    Always includes model accuracy metrics for transparency.
    """
    try:
        from backend.ml.predictor import PredictionEngine

        engine = PredictionEngine()
        result = engine.predict(symbol.upper())
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/{symbol}/train")
async def train_model(
    symbol: str,
    horizon: int = Query(5, description="Prediction horizon in days: 1, 5, or 20"),
):
    """
    Train ML models for a specific stock. This may take a few minutes.
    Uses walk-forward validation to prevent look-ahead bias.
    """
    try:
        from backend.ml.trainer import ModelTrainer

        trainer = ModelTrainer()
        result = trainer.train_symbol(symbol.upper(), horizon=horizon)
        return {
            "symbol": symbol.upper(),
            "horizon_days": horizon,
            "training_result": result,
            "disclaimer": settings.disclaimer,
        }
    except Exception as e:
        logger.error(f"Training error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/{symbol}/features")
async def get_features(symbol: str):
    """Get computed ML features for a stock (useful for debugging and transparency)."""
    try:
        from backend.ml.features import FeatureEngineer
        from backend.data.fetcher import StockDataFetcher

        df = StockDataFetcher.get_ohlcv(symbol, period="1y")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        engineer = FeatureEngineer()
        features_df = engineer.compute_features(df)

        # Return last 5 rows of features
        feature_names = engineer.get_feature_names()
        latest = features_df[feature_names].tail(5)

        return {
            "symbol": symbol.upper(),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "latest_features": latest.to_dict("records"),
            "disclaimer": settings.disclaimer,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Features error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
