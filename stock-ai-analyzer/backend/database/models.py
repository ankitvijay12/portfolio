"""
SQLAlchemy ORM models for the Stock AI Analyzer database.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, date

Base = declarative_base()


class Stock(Base):
    """Master stock table with basic info and sector mapping."""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)  # e.g. "RELIANCE"
    yf_symbol = Column(String(25), nullable=False)  # e.g. "RELIANCE.NS"
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(200))
    market_cap = Column(Float)
    exchange = Column(String(10), default="NSE")  # NSE or BSE
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    prices = relationship("DailyPrice", back_populates="stock", cascade="all, delete-orphan")
    signals = relationship("Signal", back_populates="stock", cascade="all, delete-orphan")


class DailyPrice(Base):
    """Daily OHLCV price data."""
    __tablename__ = "daily_prices"
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_stock_date"),
        Index("ix_stock_date", "stock_id", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    delivery_pct = Column(Float)  # Delivery percentage if available

    stock = relationship("Stock", back_populates="prices")


class Signal(Base):
    """Stores generated analysis signals with confidence scores."""
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signal_stock_date", "stock_id", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    signal_type = Column(String(50), nullable=False)  # "TECHNICAL", "FUNDAMENTAL", "ML", "SENTIMENT"
    direction = Column(String(10))  # "BUY", "SELL", "NEUTRAL"
    confidence = Column(Float)  # 0-100
    details = Column(JSON)  # Detailed indicator values
    created_at = Column(DateTime, default=datetime.utcnow)

    stock = relationship("Stock", back_populates="signals")


class TradePlanRecord(Base):
    """Stores generated trade plans for tracking."""
    __tablename__ = "trade_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False)
    trade_type = Column(String(20))  # "INTRADAY", "SWING", "POSITIONAL"
    direction = Column(String(10))  # "BUY", "SELL", "NO_TRADE"
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_1 = Column(Float)
    target_2 = Column(Float)
    target_3 = Column(Float)
    risk_reward = Column(Float)
    confidence = Column(Float)
    position_size = Column(Integer)  # Number of shares
    capital_required = Column(Float)
    plan_details = Column(JSON)
    outcome = Column(String(20))  # "WIN", "LOSS", "OPEN", "EXPIRED"
    pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    """Stores backtesting results for strategies."""
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    symbol = Column(String(20))
    timeframe = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    total_return_pct = Column(Float)
    params = Column(JSON)
    equity_curve = Column(JSON)  # List of equity values
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchlistItem(Base):
    """User watchlist entries with alert conditions."""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    alert_buy_zone = Column(Float)
    alert_stop_loss = Column(Float)
    alert_target = Column(Float)
    notes = Column(Text)
    is_active = Column(Boolean, default=True)


class TradeJournalRecord(Base):
    """User trade journal entries for live and past short-term trades."""
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    setup_name = Column(String(100), nullable=False)
    mode = Column(String(20), default="SWING")  # "INTRADAY", "SWING", "SHORT_TERM"
    direction = Column(String(10), default="BUY")
    entry_zone_min = Column(Float)
    entry_zone_max = Column(Float)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target_1 = Column(Float)
    target_2 = Column(Float)
    target_3 = Column(Float)
    time_stop_days = Column(Integer, default=5)
    quantity = Column(Integer, default=0)
    capital_allocated = Column(Float, default=0.0)
    risk_amount = Column(Float, default=0.0)
    risk_reward = Column(Float)
    confidence = Column(Float)
    status = Column(String(30), default="OPEN")  # "OPEN", "T1_HIT", "T2_HIT", "T3_HIT", "SL_HIT", "TIME_STOP_EXIT", "CLOSED", "CANCELLED"
    entry_date = Column(Date, default=date.today)
    exit_date = Column(Date, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

