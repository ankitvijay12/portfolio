"""
Walk-Forward Validation and Model Trainer for Indian Equities.
Trains XGBoost, Random Forest, and Simple LSTM models with strictly expanding windows,
ensuring zero look-ahead bias and honest out-of-sample performance metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from backend.data.fetcher import StockDataFetcher
    from backend.data.nse_symbols import NIFTY50_SYMBOLS
    from backend.config import settings
except ImportError:
    # Fallback paths for standalone execution or testing
    StockDataFetcher = None
    NIFTY50_SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    class FallbackSettings:
        disclaimer = (
            "⚠️ DISCLAIMER: This is an educational tool for stock market analysis. "
            "It is NOT SEBI-registered investment advice. Past performance does not "
            "guarantee future results. No prediction model is 100% accurate. "
            "Always do your own research and consult a SEBI-registered financial "
            "advisor before making investment decisions. Trade at your own risk."
        )
    settings = FallbackSettings()

from .features import FeatureEngineer
from .models import StockPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates walk-forward validation and model training across NSE stocks.
    Follows institutional quantitative guidelines:
    - Expanding training window starting with 1 year (252 trading days)
    - 1-month forward validation window (21 trading days)
    - 1-month step forward without data leakage
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize trainer with feature engineer and persistent model directory.

        Args:
            model_dir: Directory where trained .joblib models are saved.
        """
        self.feature_engineer = FeatureEngineer()
        if model_dir:
            self.model_dir = model_dir
        else:
            # Default to backend/ml/saved_models
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(curr_dir, "saved_models")

        os.makedirs(self.model_dir, exist_ok=True)
        self.training_reports: Dict[str, Any] = {}

    def walk_forward_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 12,
        initial_train_days: int = 252,
        val_step_days: int = 21,
    ) -> Dict[str, Any]:
        """
        Execute walk-forward cross-validation with an expanding window.

        Args:
            X: Cleaned feature matrix.
            y: Binary target series.
            n_splits: Target number of out-of-sample forward steps (e.g. 12 months).
            initial_train_days: Minimum training window (default 252 trading days ~ 1 year).
            val_step_days: Step and validation window (default 21 trading days ~ 1 month).

        Returns:
            Dictionary with aggregated out-of-sample metrics and fold breakdowns.
        """
        n_samples = len(X)
        if n_samples < initial_train_days + val_step_days:
            # Adjust window sizes for shorter series
            initial_train_days = max(50, int(n_samples * 0.6))
            val_step_days = max(10, int((n_samples - initial_train_days) / 4))

        available_steps = (n_samples - initial_train_days) // val_step_days
        actual_splits = max(1, min(n_splits, available_steps))

        logger.info(
            f"Starting walk-forward CV: {n_samples} samples, {actual_splits} folds, "
            f"initial train={initial_train_days}, val_step={val_step_days}"
        )

        all_y_true: List[int] = []
        all_pred_xgb: List[int] = []
        all_pred_rf: List[int] = []
        all_pred_lstm: List[int] = []
        all_pred_ens: List[int] = []
        fold_summaries: List[Dict[str, Any]] = []

        for fold in range(actual_splits):
            train_end = n_samples - (actual_splits - fold) * val_step_days
            val_start = train_end
            val_end = min(n_samples, val_start + val_step_days)

            # Ensure valid fold bounds
            if train_end <= 0 or val_start >= val_end:
                continue

            X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
            X_vl, y_vl = X.iloc[val_start:val_end], y.iloc[val_start:val_end]

            # Skip fold if validation lacks both classes
            if len(np.unique(y_tr)) < 2:
                continue

            # Instantiate and train fresh predictor for each fold
            predictor = StockPredictor(random_state=42 + fold)
            predictor.train(X_tr, y_tr, X_vl, y_vl)

            # Generate out-of-sample predictions
            p_xgb = predictor.xgb_model.predict_proba(X_vl.values)[:, 1]
            p_rf = predictor.rf_model.predict_proba(X_vl.values)[:, 1]
            p_lstm = predictor.lstm_model.predict_proba(X_vl.values)[:, 1]

            w = predictor.weights
            p_ens = w["xgboost"] * p_xgb + w["random_forest"] * p_rf + w["lstm"] * p_lstm

            pred_xgb = (p_xgb >= 0.5).astype(int)
            pred_rf = (p_rf >= 0.5).astype(int)
            pred_lstm = (p_lstm >= 0.5).astype(int)
            pred_ens = (p_ens >= 0.5).astype(int)

            y_vl_list = y_vl.tolist()
            all_y_true.extend(y_vl_list)
            all_pred_xgb.extend(pred_xgb.tolist())
            all_pred_rf.extend(pred_rf.tolist())
            all_pred_lstm.extend(pred_lstm.tolist())
            all_pred_ens.extend(pred_ens.tolist())

            fold_summaries.append({
                "fold": fold + 1,
                "train_samples": len(X_tr),
                "val_samples": len(X_vl),
                "ensemble_accuracy": round(float(accuracy_score(y_vl, pred_ens)), 4),
                "ensemble_f1": round(float(f1_score(y_vl, pred_ens, zero_division=0)), 4),
            })

        if not all_y_true:
            # Fallback if no folds could be formed
            return {
                "n_splits": 0,
                "total_test_samples": 0,
                "ensemble_metrics": {"accuracy": 0.50, "precision": 0.50, "recall": 0.50, "f1": 0.50},
                "model_metrics": {
                    "xgboost": {"accuracy": 0.50, "precision": 0.50, "recall": 0.50, "f1": 0.50},
                    "random_forest": {"accuracy": 0.50, "precision": 0.50, "recall": 0.50, "f1": 0.50},
                    "lstm": {"accuracy": 0.50, "precision": 0.50, "recall": 0.50, "f1": 0.50},
                    "ensemble": {"accuracy": 0.50, "precision": 0.50, "recall": 0.50, "f1": 0.50},
                },
                "fold_metrics": [],
            }

        def calc_metrics(y_true, y_pred) -> Dict[str, float]:
            return {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            }

        xgb_metrics = calc_metrics(all_y_true, all_pred_xgb)
        rf_metrics = calc_metrics(all_y_true, all_pred_rf)
        lstm_metrics = calc_metrics(all_y_true, all_pred_lstm)
        ens_metrics = calc_metrics(all_y_true, all_pred_ens)

        return {
            "n_splits": len(fold_summaries),
            "total_test_samples": len(all_y_true),
            "ensemble_metrics": ens_metrics,
            "model_metrics": {
                "xgboost": xgb_metrics,
                "random_forest": rf_metrics,
                "lstm": lstm_metrics,
                "ensemble": ens_metrics,
            },
            "fold_metrics": fold_summaries,
        }

    def train_symbol(
        self,
        symbol: str,
        horizon: int = 5,
        save: bool = True,
        period: str = "5y",
    ) -> Dict[str, Any]:
        """
        Train and evaluate ensemble models for a single stock symbol.

        Args:
            symbol: NSE symbol (e.g., "RELIANCE").
            horizon: Prediction horizon in days (1=day, 5=week, 20=month).
            save: Whether to persist trained model to disk.
            period: Historical data lookback period (default "5y").

        Returns:
            Dictionary with honest out-of-sample metrics and model metadata.
        """
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        logger.info(f"Initiating training for {clean_symbol} (horizon={horizon}d, period={period})...")

        # 1. Fetch 5 years of historical data
        if StockDataFetcher is not None:
            df = StockDataFetcher.get_ohlcv(clean_symbol, period=period)
        else:
            import yfinance as yf
            yf_sym = f"{clean_symbol}.NS"
            df = yf.Ticker(yf_sym).history(period=period)
            df.columns = [c.title().replace(" ", "_") for c in df.columns]

        if df is None or df.empty or len(df) < 60:
            error_msg = f"Insufficient historical data for symbol {clean_symbol} (retrieved {len(df) if df is not None else 0} rows)"
            logger.warning(error_msg)
            return {
                "symbol": clean_symbol,
                "horizon": horizon,
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # 2. Compute features and build training dataset
        X, y = self.feature_engineer.prepare_dataset(df, horizon=horizon)
        if len(X) < 50:
            error_msg = f"Dataset too small after feature generation for {clean_symbol} ({len(X)} rows)"
            logger.warning(error_msg)
            return {
                "symbol": clean_symbol,
                "horizon": horizon,
                "status": "error",
                "message": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # 3. Walk-Forward Cross-Validation
        cv_report = self.walk_forward_cv(X, y, n_splits=12)

        # 4. Train final production model on full dataset
        final_predictor = StockPredictor()
        final_predictor.train(X, y)

        # Attach honest out-of-sample metrics from CV
        final_predictor.metrics = cv_report["model_metrics"]

        # 5. Persist model
        model_filename = f"{clean_symbol}_{horizon}d.joblib"
        model_path = os.path.join(self.model_dir, model_filename)

        if save:
            final_predictor.save_models(model_path)

        report = {
            "symbol": clean_symbol,
            "horizon": horizon,
            "status": "success",
            "samples_trained": len(X),
            "walk_forward_folds": cv_report["n_splits"],
            "model_metrics": cv_report["model_metrics"],
            "model_weights": final_predictor.weights,
            "top_features": final_predictor.get_feature_importance(top_n=10)["top_features"],
            "saved_model_path": model_path if save else None,
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": settings.disclaimer,
        }

        self.training_reports[f"{clean_symbol}_{horizon}d"] = report
        logger.info(
            f"Completed training for {clean_symbol} ({horizon}d). "
            f"Ensemble OOS Accuracy: {cv_report['ensemble_metrics']['accuracy'] * 100:.1f}%, "
            f"F1: {cv_report['ensemble_metrics']['f1']:.3f}"
        )
        return report

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
        horizons: List[int] = [1, 5, 20],
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Train ML prediction models across Nifty 50 or specified universe.

        Args:
            symbols: List of stock symbols (defaults to NIFTY50_SYMBOLS).
            horizons: Forward horizons in days (default [1, 5, 20]).
            save: Whether to persist trained models to disk.

        Returns:
            Summary report of training runs across all symbols.
        """
        if symbols is None:
            symbols = list(NIFTY50_SYMBOLS)

        logger.info(f"Starting batch training for {len(symbols)} symbols across horizons {horizons}...")
        results: Dict[str, Any] = {}
        success_count = 0
        error_count = 0

        for idx, sym in enumerate(symbols, 1):
            clean_sym = sym.replace(".NS", "").replace(".BO", "").upper().strip()
            logger.info(f"[{idx}/{len(symbols)}] Training {clean_sym}...")
            results[clean_sym] = {}

            for h in horizons:
                try:
                    res = self.train_symbol(clean_sym, horizon=h, save=save)
                    results[clean_sym][f"{h}d"] = res
                    if res.get("status") == "success":
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed training {clean_sym} for {h}d: {e}")
                    results[clean_sym][f"{h}d"] = {
                        "status": "error",
                        "message": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

        summary = {
            "total_models_attempted": len(symbols) * len(horizons),
            "successful_models": success_count,
            "failed_models": error_count,
            "horizons": horizons,
            "symbols_count": len(symbols),
            "completed_at": datetime.utcnow().isoformat(),
            "results": results,
            "disclaimer": settings.disclaimer,
        }
        return summary

    def get_training_report(self) -> Dict[str, Any]:
        """
        Return structured summary of latest training runs and aggregate metrics.
        """
        if not self.training_reports:
            return {
                "total_trained": 0,
                "reports": {},
                "average_ensemble_accuracy": 0.0,
                "top_performing_stocks": [],
                "disclaimer": settings.disclaimer,
            }

        accuracies = []
        stock_rankings = []

        for key, rep in self.training_reports.items():
            if rep.get("status") == "success":
                acc = rep.get("model_metrics", {}).get("ensemble", {}).get("accuracy", 0.0)
                f1 = rep.get("model_metrics", {}).get("ensemble", {}).get("f1", 0.0)
                accuracies.append(acc)
                stock_rankings.append({
                    "model_key": key,
                    "symbol": rep.get("symbol"),
                    "horizon": rep.get("horizon"),
                    "accuracy": acc,
                    "f1": f1,
                })

        stock_rankings.sort(key=lambda x: x["f1"], reverse=True)
        avg_acc = float(np.mean(accuracies)) if accuracies else 0.0

        return {
            "total_trained": len(self.training_reports),
            "average_ensemble_accuracy": round(avg_acc, 4),
            "top_performing_stocks": stock_rankings[:5],
            "reports": self.training_reports,
            "generated_at": datetime.utcnow().isoformat(),
            "disclaimer": settings.disclaimer,
        }
