"""
APScheduler-based daily data refresh, periodic screener refresh, and model retraining scheduler.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import logging
import threading
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def refresh_stock_data():
    """Daily job: refresh OHLCV data for all tracked stocks."""
    from backend.data.nse_symbols import NIFTY50_SYMBOLS
    from backend.data.fetcher import StockDataFetcher

    logger.info(f"Starting daily data refresh at {datetime.now()}")
    success = 0
    failed = 0

    for symbol in NIFTY50_SYMBOLS:
        try:
            # Clear cache to force fresh fetch
            StockDataFetcher._cache.clear()
            StockDataFetcher._cache_expiry.clear()

            df = StockDataFetcher.get_ohlcv(symbol, period="5d")
            if not df.empty:
                success += 1
            else:
                failed += 1
                logger.warning(f"No data for {symbol}")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to refresh {symbol}: {e}")

    logger.info(f"Data refresh complete: {success} success, {failed} failed")


def refresh_screener_job():
    """Periodic job: refresh stock screener cache every 15 minutes in background."""
    try:
        from backend.routers.screener import refresh_screener_cache
        refresh_screener_cache()
    except Exception as e:
        logger.error(f"Screener background refresh failed: {e}")


def refresh_short_term_scanner_job():
    """Periodic job: refresh short-term scanner cache every 8 minutes in background."""
    try:
        from backend.trading.short_term_scanner import short_term_scanner
        short_term_scanner.scan_all(mode="SWING", force_refresh=True)
    except Exception as e:
        logger.error(f"Short-term scanner background refresh failed: {e}")


def retrain_models():
    """Weekly job: retrain ML models with latest data."""
    logger.info(f"Starting model retraining at {datetime.now()}")
    try:
        from backend.ml.trainer import ModelTrainer
        trainer = ModelTrainer()
        trainer.train_all()
        logger.info("Model retraining complete")
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")


def start_scheduler():
    """Initialize and start the background scheduler."""
    # 1. Daily data refresh at configured time (default 6 PM IST)
    scheduler.add_job(
        refresh_stock_data,
        trigger=CronTrigger(
            hour=settings.data_refresh_hour,
            minute=settings.data_refresh_minute,
            timezone="Asia/Kolkata",
        ),
        id="daily_refresh",
        name="Daily Stock Data Refresh",
        replace_existing=True,
    )

    # 2. 15-minute periodic screener cache refresh
    scheduler.add_job(
        refresh_screener_job,
        trigger=IntervalTrigger(minutes=15),
        id="screener_refresh",
        name="Periodic Screener Cache Refresh",
        replace_existing=True,
    )

    # 3. 8-minute periodic short-term scanner cache refresh
    scheduler.add_job(
        refresh_short_term_scanner_job,
        trigger=IntervalTrigger(minutes=8),
        id="short_term_scanner_refresh",
        name="Periodic Short-Term Scanner Cache Refresh",
        replace_existing=True,
    )

    # 4. Weekly model retraining on Sunday at 2 AM IST
    scheduler.add_job(
        retrain_models,
        trigger=CronTrigger(
            day_of_week="sun",
            hour=2,
            minute=0,
            timezone="Asia/Kolkata",
        ),
        id="weekly_retrain",
        name="Weekly Model Retraining",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with daily refresh, 15m screener refresh, 8m short-term scan, and weekly retraining jobs")

    # Prime caches asynchronously in background threads so first request is instant
    threading.Thread(target=refresh_screener_job, daemon=True).start()
    threading.Thread(target=refresh_short_term_scanner_job, daemon=True).start()



def stop_scheduler():
    """Shut down the scheduler gracefully."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
