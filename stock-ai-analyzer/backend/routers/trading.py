"""
Trading API routes — Trade plans, position sizing, and watchlist management.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Optional
from pydantic import BaseModel
import logging
from datetime import date

from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trading", tags=["Trading"])


class WatchlistRequest(BaseModel):
    symbol: str
    alert_buy_zone: Optional[float] = None
    alert_stop_loss: Optional[float] = None
    alert_target: Optional[float] = None
    notes: Optional[str] = None


class PositionSizeRequest(BaseModel):
    capital: float = 1000000
    risk_pct: float = 1.0
    entry_price: float
    stop_loss: float


@router.get("/trade-plan/{symbol}")
async def get_trade_plan(
    symbol: str,
    trade_type: str = Query("SWING", description="INTRADAY, SWING, or POSITIONAL"),
    capital: float = Query(1000000, description="Trading capital in INR"),
    risk_pct: float = Query(1.0, description="Risk percentage per trade"),
):
    """
    Generate a complete trade plan with entry, SL, targets, and position sizing.
    Includes a NO_TRADE recommendation when the setup is weak.
    """
    try:
        from backend.trading.trade_plan import TradePlanGenerator

        generator = TradePlanGenerator()
        plan = generator.generate(
            symbol=symbol.upper(),
            trade_type=trade_type.upper(),
            capital=capital,
            risk_pct=risk_pct,
        )

        plan["disclaimer"] = settings.disclaimer
        return plan

    except Exception as e:
        logger.error(f"Trade plan error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Trade plan generation failed: {str(e)}")


@router.get("/trade-plan/{symbol}/all")
async def get_all_trade_plans(symbol: str):
    """Generate trade plans for all timeframes: Intraday, Swing, Positional."""
    try:
        from backend.trading.trade_plan import TradePlanGenerator

        generator = TradePlanGenerator()
        plans = generator.generate_all_types(symbol.upper())
        return {"symbol": symbol.upper(), "plans": plans, "disclaimer": settings.disclaimer}
    except Exception as e:
        logger.error(f"All trade plans error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position-size")
async def calculate_position_size(request: PositionSizeRequest):
    """Calculate position size based on capital, risk, entry, and stop-loss."""
    try:
        from backend.trading.position_sizing import PositionSizer

        sizer = PositionSizer()
        result = sizer.calculate(
            capital=request.capital,
            risk_pct=request.risk_pct,
            entry=request.entry_price,
            stop_loss=request.stop_loss,
        )
        return result
    except Exception as e:
        logger.error(f"Position size error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist():
    """Get all watchlist items."""
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import WatchlistItem

        db = SessionLocal()
        try:
            items = db.query(WatchlistItem).filter(WatchlistItem.is_active == True).all()
            return {
                "items": [
                    {
                        "id": item.id,
                        "symbol": item.symbol,
                        "added_at": item.added_at.isoformat() if item.added_at else None,
                        "alert_buy_zone": item.alert_buy_zone,
                        "alert_stop_loss": item.alert_stop_loss,
                        "alert_target": item.alert_target,
                        "notes": item.notes,
                    }
                    for item in items
                ]
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Watchlist fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist")
async def add_to_watchlist(request: WatchlistRequest):
    """Add a stock to the watchlist."""
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import WatchlistItem

        db = SessionLocal()
        try:
            item = WatchlistItem(
                symbol=request.symbol.upper(),
                alert_buy_zone=request.alert_buy_zone,
                alert_stop_loss=request.alert_stop_loss,
                alert_target=request.alert_target,
                notes=request.notes,
            )
            db.add(item)
            db.commit()
            db.refresh(item)
            return {"message": f"{request.symbol.upper()} added to watchlist", "id": item.id}
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Watchlist add error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/watchlist/{item_id}")
async def remove_from_watchlist(item_id: int):
    """Remove a stock from the watchlist."""
    try:
        from backend.database.connection import SessionLocal
        from backend.database.models import WatchlistItem

        db = SessionLocal()
        try:
            item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
            if not item:
                raise HTTPException(status_code=404, detail="Watchlist item not found")
            item.is_active = False
            db.commit()
            return {"message": "Removed from watchlist"}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Watchlist remove error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
