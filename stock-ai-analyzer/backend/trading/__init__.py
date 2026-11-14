"""
Trading Module for Stock AI Analyzer.

Provides trade plan generation, position sizing, risk management,
and strategy backtesting for Indian equity markets.
"""
try:
    import importlib.metadata
except Exception:
    pass

from .trade_plan import TradePlanGenerator
from .position_sizing import PositionSizer
from .backtester import StrategyBacktester
from .short_term_scanner import ShortTermScanner, short_term_scanner
from .tax_calculator import CapitalGainsTaxCalculator

__all__ = [
    "TradePlanGenerator",
    "PositionSizer",
    "StrategyBacktester",
    "ShortTermScanner",
    "short_term_scanner",
    "CapitalGainsTaxCalculator",
]

