"""
Position Sizing Module for Stock AI Analyzer.

Calculates optimal capital allocation and risk exposure for trades
using Fixed Fractional Risk and Kelly Criterion models tailored
for Indian equity markets (NSE/BSE).
"""

from typing import Dict, Any, Optional
import math
import logging
from backend.config import settings

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Computes position sizes, capital required, and capital at risk.
    Guarantees strict compliance with portfolio risk management rules.
    """

    def __init__(self, default_capital: Optional[float] = None, default_risk_pct: Optional[float] = None):
        """
        Initialize position sizer with capital and risk settings.

        Args:
            default_capital: Total account capital in INR (default: from settings)
            default_risk_pct: Maximum percentage of capital to risk per trade (default: from settings)
        """
        self.default_capital = default_capital or getattr(settings, "default_capital", 1000000.0)
        self.default_risk_pct = default_risk_pct or getattr(settings, "default_risk_pct", 1.0)

    def fixed_fractional(
        self,
        capital: float,
        risk_pct: float,
        entry: float,
        stop_loss: float,
    ) -> Dict[str, Any]:
        """
        Calculate position size based on Fixed Fractional Risk model.

        Risk per trade is capped at a fixed percentage of total capital.
        Quantity = (Capital * Risk%) / |Entry - StopLoss|

        Args:
            capital: Total trading capital in INR
            risk_pct: Maximum risk percentage (e.g., 1.0 for 1%)
            entry: Proposed entry price in INR
            stop_loss: Stop-loss price in INR

        Returns:
            Dictionary with quantity, capital_required, capital_at_risk,
            risk_per_share, risk_pct_of_capital, and confidence score.
        """
        try:
            capital = float(capital)
            risk_pct = float(risk_pct)
            entry = float(entry)
            stop_loss = float(stop_loss)

            if capital <= 0 or entry <= 0:
                logger.warning("Invalid capital or entry price provided to fixed_fractional")
                return {
                    "quantity": 0,
                    "capital_required": 0.0,
                    "capital_at_risk": 0.0,
                    "risk_per_share": 0.0,
                    "risk_pct_of_capital": 0.0,
                    "confidence": 0.0,
                    "error": "Capital and entry price must be positive numbers",
                    "disclaimer": settings.disclaimer,
                }

            risk_per_share = abs(entry - stop_loss)
            if risk_per_share <= 0:
                logger.warning("Stop loss equals entry price; risk per share is zero")
                return {
                    "quantity": 0,
                    "capital_required": 0.0,
                    "capital_at_risk": 0.0,
                    "risk_per_share": 0.0,
                    "risk_pct_of_capital": 0.0,
                    "confidence": 0.0,
                    "error": "Stop loss must not equal entry price",
                    "disclaimer": settings.disclaimer,
                }

            # Maximum amount allowed to lose on this trade
            capital_risk_budget = capital * (risk_pct / 100.0)

            # Share count from risk tolerance
            shares_by_risk = math.floor(capital_risk_budget / risk_per_share)

            # Cash affordability constraint (no unhedged margin/overnight debt)
            max_shares_affordable = math.floor(capital / entry)

            quantity = int(max(0, min(shares_by_risk, max_shares_affordable)))
            capital_required = round(quantity * entry, 2)
            capital_at_risk = round(quantity * risk_per_share, 2)
            risk_pct_of_capital = round((capital_at_risk / capital) * 100.0, 2) if capital > 0 else 0.0

            # Confidence score for sizing calculation (100% when mathematically sound)
            confidence = 100.0 if quantity > 0 else 30.0

            return {
                "quantity": quantity,
                "capital_required": capital_required,
                "capital_at_risk": capital_at_risk,
                "risk_per_share": round(risk_per_share, 2),
                "risk_pct_of_capital": risk_pct_of_capital,
                "confidence": confidence,
                "method": "fixed_fractional",
                "disclaimer": settings.disclaimer,
            }

        except Exception as e:
            logger.error(f"Error in fixed_fractional position sizing: {e}", exc_info=True)
            return {
                "quantity": 0,
                "capital_required": 0.0,
                "capital_at_risk": 0.0,
                "risk_per_share": 0.0,
                "risk_pct_of_capital": 0.0,
                "confidence": 0.0,
                "error": str(e),
                "disclaimer": settings.disclaimer,
            }

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fractional_multiplier: float = 0.5,
    ) -> float:
        """
        Calculate optimal position size as a fraction of capital using the Kelly Criterion.

        Formula: f* = p - (1 - p) / b = (p*(b + 1) - 1) / b
        where:
            p = win rate (probability of winning)
            b = payoff ratio (avg_win / avg_loss)

        A fractional multiplier (default: 0.5 for Half-Kelly) is applied for risk conservatism.

        Args:
            win_rate: Historical win rate (0.0 to 1.0 or 0% to 100%)
            avg_win: Average profit on winning trades in INR or percentage
            avg_loss: Average loss on losing trades in INR or percentage
            fractional_multiplier: Kelly scaling factor (0.5 for Half-Kelly)

        Returns:
            Optimal allocation as fraction of capital (0.0 to 0.25 max single stock cap)
        """
        try:
            # Normalize win rate if given as percentage 0-100
            p = win_rate / 100.0 if win_rate > 1.0 else win_rate
            if p <= 0.0 or p >= 1.0:
                p = max(0.01, min(0.99, p))

            loss = abs(avg_loss)
            win = abs(avg_win)
            if loss <= 0.0 or win <= 0.0:
                return 0.0

            b = win / loss  # Win/Loss ratio
            if b <= 0.0:
                return 0.0

            q = 1.0 - p
            # Kelly formula: f* = p - (q / b)
            f_star = p - (q / b)

            if f_star <= 0.0:
                # Negative statistical edge; no capital should be risked
                return 0.0

            # Scale by fractional multiplier (e.g., Half-Kelly) and cap allocation at 25% of capital
            adjusted_fraction = f_star * fractional_multiplier
            safe_fraction = max(0.0, min(adjusted_fraction, 0.25))

            return round(safe_fraction, 4)

        except Exception as e:
            logger.error(f"Error in kelly_criterion calculation: {e}", exc_info=True)
            return 0.0

    def calculate(
        self,
        capital: Optional[float] = None,
        risk_pct: Optional[float] = None,
        entry: float = 0.0,
        stop_loss: float = 0.0,
        method: str = "fixed_fractional",
        win_rate: float = 0.50,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Main entry point for position sizing.

        Args:
            capital: Total account capital in INR (default: default_capital)
            risk_pct: Percentage risk per trade (default: default_risk_pct)
            entry: Proposed entry price (alias 'entry_price' also accepted)
            stop_loss: Stop loss price
            method: Sizing method ('fixed_fractional' or 'kelly')
            win_rate: Strategy win rate for Kelly calculation
            avg_win: Average win for Kelly calculation
            avg_loss: Average loss for Kelly calculation
            kwargs: Extra parameters (e.g., 'entry_price')

        Returns:
            Dictionary matching the standard format:
            {
                "quantity": 42,
                "capital_required": 100800.0,
                "capital_at_risk": 1680.0,
                "risk_per_share": 40.0,
                "risk_pct_of_capital": 1.0,
                "confidence": 100.0,
                ...
            }
        """
        # Support alias entry_price
        if entry <= 0.0 and "entry_price" in kwargs:
            entry = float(kwargs["entry_price"])

        cap = capital if capital is not None and capital > 0 else self.default_capital
        r_pct = risk_pct if risk_pct is not None and risk_pct > 0 else self.default_risk_pct

        method_clean = method.lower().strip()

        if method_clean in ("kelly", "kelly_criterion"):
            kelly_frac = self.kelly_criterion(win_rate, avg_win, avg_loss)
            risk_per_share = abs(entry - stop_loss)

            if kelly_frac <= 0.0 or entry <= 0.0 or risk_per_share <= 0.0:
                return {
                    "quantity": 0,
                    "capital_required": 0.0,
                    "capital_at_risk": 0.0,
                    "risk_per_share": round(risk_per_share, 2),
                    "risk_pct_of_capital": 0.0,
                    "kelly_fraction": 0.0,
                    "confidence": 30.0,
                    "method": "kelly",
                    "disclaimer": settings.disclaimer,
                }

            allocated_capital = cap * kelly_frac
            max_shares_affordable = math.floor(allocated_capital / entry)
            quantity = int(max(0, max_shares_affordable))

            capital_required = round(quantity * entry, 2)
            capital_at_risk = round(quantity * risk_per_share, 2)
            risk_pct_of_capital = round((capital_at_risk / cap) * 100.0, 2) if cap > 0 else 0.0

            return {
                "quantity": quantity,
                "capital_required": capital_required,
                "capital_at_risk": capital_at_risk,
                "risk_per_share": round(risk_per_share, 2),
                "risk_pct_of_capital": risk_pct_of_capital,
                "kelly_fraction": kelly_frac,
                "confidence": 90.0,
                "method": "kelly",
                "disclaimer": settings.disclaimer,
            }

        # Default: fixed fractional model
        return self.fixed_fractional(capital=cap, risk_pct=r_pct, entry=entry, stop_loss=stop_loss)
