"""
Backtesting API routes.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import logging

from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/backtest", tags=["Backtesting"])


class BacktestRequest(BaseModel):
    symbol: str = "RELIANCE"
    strategy: str = "ema_crossover"  # ema_crossover, rsi_reversal, supertrend, breakout, macd
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    capital: float = 1000000
    fast_ema: int = 20
    slow_ema: int = 50
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70


@router.post("/run")
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest for a given strategy and stock.
    Includes realistic costs: brokerage (₹20/trade), STT (0.1%), slippage (0.05%).
    """
    try:
        from backend.trading.backtester import StrategyBacktester

        backtester = StrategyBacktester()
        result = backtester.backtest(
            symbol=request.symbol.upper(),
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            capital=request.capital,
        )
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/strategies")
async def list_strategies():
    """List all available backtesting strategies."""
    return {
        "strategies": [
            {
                "id": "ema_crossover",
                "name": "EMA Crossover",
                "description": "Buy when 20 EMA crosses above 50 EMA, sell on cross below.",
                "params": ["fast_ema", "slow_ema"],
            },
            {
                "id": "rsi_reversal",
                "name": "RSI Reversal",
                "description": "Buy when RSI crosses above 30 (oversold), sell when crosses below 70.",
                "params": ["rsi_period", "rsi_oversold", "rsi_overbought"],
            },
            {
                "id": "supertrend",
                "name": "Supertrend",
                "description": "Buy/sell based on Supertrend indicator direction changes.",
                "params": ["period", "multiplier"],
            },
            {
                "id": "breakout",
                "name": "20-Day Breakout",
                "description": "Buy when price breaks above 20-day high with volume > 1.5x average.",
                "params": ["lookback", "volume_threshold"],
            },
            {
                "id": "macd",
                "name": "MACD Crossover",
                "description": "Buy when MACD histogram turns positive, sell when negative.",
                "params": ["fast", "slow", "signal"],
            },
        ],
        "disclaimer": settings.disclaimer,
    }


@router.get("/run-all/{symbol}")
async def run_all_backtests(
    symbol: str,
    capital: float = Query(1000000, description="Initial capital"),
):
    """Run all strategies on a stock and compare results."""
    try:
        from backend.trading.backtester import StrategyBacktester

        backtester = StrategyBacktester()
        results = backtester.backtest_all_strategies(symbol.upper(), capital=capital)
        return {
            "symbol": symbol.upper(),
            "results": results,
            "disclaimer": settings.disclaimer,
        }
    except Exception as e:
        logger.error(f"All backtests error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
