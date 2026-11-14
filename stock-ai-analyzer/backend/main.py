"""
Stock AI Analyzer — FastAPI Application Entry Point.

A comprehensive AI-powered stock analysis platform for NSE/BSE Indian stocks.
Provides technical analysis, fundamental data, sentiment analysis, ML predictions,
trade plans, backtesting, and an interactive dashboard.

⚠️ DISCLAIMER: This is an educational tool, NOT SEBI-registered investment advice.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging
import sys

from backend.config import settings
from backend.database.connection import engine
from backend.database.models import Base

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Stock AI Analyzer...")

    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created/verified")

    # Start scheduler for daily data refresh & screener background cache
    try:
        from backend.data.scheduler import start_scheduler
        start_scheduler()
        logger.info("✅ Scheduler started")
    except Exception as e:
        logger.warning(f"⚠️ Scheduler failed to start: {e}")

    logger.info("✅ Stock AI Analyzer is ready!")
    logger.info(f"📊 API docs at http://{settings.app_host}:{settings.app_port}/docs")

    yield

    # Shutdown
    logger.info("Shutting down Stock AI Analyzer...")
    try:
        from backend.data.scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass
    logger.info("👋 Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Stock AI Analyzer",
    description=(
        "AI-powered stock analysis platform for NSE/BSE Indian stocks. "
        "Provides technical analysis, fundamental data, sentiment analysis, "
        "ML predictions, trade plans, and backtesting.\n\n"
        f"**{settings.disclaimer}**"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# GZip compression middleware (compresses responses > 1000 bytes)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from backend.routers.analysis import router as analysis_router
from backend.routers.screener import router as screener_router
from backend.routers.trading import router as trading_router
from backend.routers.ml import router as ml_router
from backend.routers.backtest import router as backtest_router
from backend.routers.short_term import router as short_term_router

app.include_router(analysis_router)
app.include_router(screener_router)
app.include_router(trading_router)
app.include_router(ml_router)
app.include_router(backtest_router)
app.include_router(short_term_router)


@app.get("/")
async def root():
    """Root endpoint with app info and links."""
    return {
        "app": "Stock AI Analyzer",
        "version": "1.0.0",
        "description": "AI-powered stock analysis for NSE/BSE Indian stocks",
        "endpoints": {
            "docs": "/docs",
            "quote": "/api/analysis/{symbol}/quote",
            "analysis": "/api/analysis/{symbol}",
            "screener": "/api/screener/scan",
            "trade_plan": "/api/trading/trade-plan/{symbol}",
            "prediction": "/api/predict/{symbol}",
            "backtest": "/api/backtest/run",
        },
        "disclaimer": settings.disclaimer,
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/disclaimer")
async def disclaimer():
    """Get the SEBI disclaimer."""
    return {"disclaimer": settings.disclaimer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
