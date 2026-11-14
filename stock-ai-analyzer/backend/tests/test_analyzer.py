"""
Unit and integration tests for Stock AI Analyzer modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.data.fetcher import StockDataFetcher
from backend.trading.position_sizing import PositionSizer
from backend.trading.backtester import StrategyBacktester


def generate_sample_ohlcv(days: int = 200) -> pd.DataFrame:
    """Helper to generate realistic synthetic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    np.random.seed(42)

    # Geometric Brownian Motion simulation
    returns = np.random.normal(loc=0.0005, scale=0.015, size=days)
    price = 2500.0 * np.exp(np.cumsum(returns))

    high = price * (1 + np.abs(np.random.normal(0, 0.008, size=days)))
    low = price * (1 - np.abs(np.random.normal(0, 0.008, size=days)))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, size=days)
    volume = np.random.randint(500000, 5000000, size=days)

    df = pd.DataFrame(
        {
            'Open': open_p,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume,
        },
        index=dates,
    )
    return df


def test_position_sizing_fixed_fractional():
    """Test 1% capital risk position sizing calculation."""
    sizer = PositionSizer()
    res = sizer.fixed_fractional(
        capital=1000000.0,
        risk_pct=1.0,
        entry=2500.0,
        stop_loss=2450.0,
    )
    # Risk amount = 1% of 1,000,000 = 10,000
    # Risk per share = 2500 - 2450 = 50
    # Quantity = 10,000 / 50 = 200 shares
    assert res['quantity'] == 200
    assert res['risk_per_share'] == 50.0
    assert res['capital_at_risk'] == 10000.0
    assert res['capital_required'] == 500000.0


def test_position_sizing_zero_or_negative_risk():
    """Test position sizing defense against invalid stop loss."""
    sizer = PositionSizer()
    res = sizer.fixed_fractional(
        capital=1000000.0,
        risk_pct=1.0,
        entry=2500.0,
        stop_loss=2500.0,  # Zero risk distance
    )
    assert res['quantity'] == 0


def test_kelly_criterion_sizing():
    """Test Kelly formula fractional sizing."""
    sizer = PositionSizer()
    kelly_f = sizer.kelly_criterion(win_rate=60.0, avg_win=4.0, avg_loss=2.0)
    # Kelly = (p*b - q) / b = (0.6*2 - 0.4)/2 = 0.8/2 = 0.4; half kelly = 0.20
    assert kelly_f > 0.0
    assert kelly_f <= 0.25  # Should respect safety ceiling


def test_patterns_recognition():
    """Test pattern recognition module against sample OHLCV."""
    from backend.analysis.patterns import PatternRecognizer

    df = generate_sample_ohlcv(120)
    recognizer = PatternRecognizer()
    results = recognizer.scan_all_patterns(df)

    assert isinstance(results, list)
    for p in results:
        assert 'pattern' in p
        assert 'direction' in p
        assert 'confidence' in p
        assert 'historical_win_rate' in p
        assert 0.0 <= p['confidence'] <= 100.0
        assert 0.0 <= p['historical_win_rate'] <= 100.0


def test_support_resistance_analysis():
    """Test S/R pivot and Fibonacci calculation."""
    from backend.analysis.support_resistance import SupportResistanceAnalyzer

    df = generate_sample_ohlcv(120)
    analyzer = SupportResistanceAnalyzer()
    res = analyzer.analyze(df)

    assert 'dynamic_sr' in res or 'confluence' in res or 'pivots' in res
    assert 'confluence' in res or 'supports' in res or 'dynamic_sr' in res


def test_backtester_execution():
    """Test backtesting engine with EMA crossover strategy."""
    backtester = StrategyBacktester()
    # Test fallback / simulation
    res = backtester.backtest(
        symbol='RELIANCE',
        strategy='ema_crossover',
        capital=1000000,
    )

    assert 'strategy' in res
    assert 'win_rate' in res
    assert 'profit_factor' in res
    assert 'max_drawdown' in res
    assert 'equity_curve' in res
