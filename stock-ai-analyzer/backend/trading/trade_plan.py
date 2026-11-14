"""
Trade Plan Generator Module for Stock AI Analyzer.

Generates complete, institutional-grade trade plans for NSE/BSE stocks
across Intraday, Swing, and Positional timeframes with confluence-based
entry zones, adaptive ATR stop-losses, multi-tier Fibonacci targets,
strict risk-reward enforcement, and comprehensive No-Trade filters.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import math
from datetime import datetime
import pandas as pd
import numpy as np

# pandas-ta indicator library
try:
    import importlib.metadata
except Exception:
    pass

try:
    import pandas_ta as ta
except Exception:
    ta = None

# Optional analysis imports with robust fallbacks
try:
    from backend.analysis.technical import TechnicalAnalyzer
except ImportError:
    TechnicalAnalyzer = None

try:
    from backend.analysis.support_resistance import SupportResistanceAnalyzer
except ImportError:
    SupportResistanceAnalyzer = None

try:
    from backend.analysis.fundamental import FundamentalAnalyzer
except ImportError:
    FundamentalAnalyzer = None

try:
    from backend.ml.predictor import PredictionEngine
except ImportError:
    PredictionEngine = None

from backend.data.fetcher import StockDataFetcher
from backend.config import settings
from backend.trading.position_sizing import PositionSizer

logger = logging.getLogger(__name__)


class TradePlanGenerator:
    """
    Generates actionable trade plans incorporating Technical, Fundamental,
    and Machine Learning confluence with strict No-Trade filtering.
    """

    def __init__(self):
        """Initialize TradePlanGenerator with position sizer and data fetcher."""
        self.sizer = PositionSizer()
        self.fetcher = StockDataFetcher

    def generate(
        self,
        symbol: str,
        trade_type: str = "SWING",
        capital: float = 1000000.0,
        risk_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive trade plan for a given stock and timeframe.

        Args:
            symbol: Stock ticker (e.g. "RELIANCE" or "TCS.NS")
            trade_type: "INTRADAY", "SWING", or "POSITIONAL"
            capital: Trading capital in INR
            risk_pct: Max risk percentage per trade

        Returns:
            Dictionary containing entry, stop-loss, targets, R:R, sizing,
            indicators, confidence score, plain English summary, and SEBI disclaimer.
        """
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
        trade_type = trade_type.upper().strip()
        if trade_type not in ("INTRADAY", "SWING", "POSITIONAL"):
            trade_type = "SWING"

        try:
            # 1. Fetch market data for appropriate timeframe
            market_data = self._fetch_market_data(clean_symbol, trade_type)
            df = market_data.get("df")

            if df is None or df.empty or len(df) < 20:
                logger.warning(f"Insufficient historical data for {clean_symbol}")
                return self._create_no_trade_plan(
                    symbol=clean_symbol,
                    trade_type=trade_type,
                    reasons=[f"Insufficient historical OHLCV data for {clean_symbol}."],
                    confidence=0.0,
                )

            # 2. Extract technical indicators and support/resistance
            technical_data = self._analyze_technicals(clean_symbol, df, trade_type)
            sr_data = self._analyze_support_resistance(clean_symbol, df)
            fundamental_data = self._analyze_fundamentals(clean_symbol)
            ml_data = self._analyze_ml_prediction(clean_symbol, df)

            # 3. Check India VIX regime
            india_vix = self._fetch_india_vix()
            vix_val = india_vix.get("value", 15.0)

            # 4. Compute Entry Zone & Confluence Confidence
            entry_price, entry_confidence = self._compute_entry_zone(technical_data, sr_data)
            current_close = float(df["Close"].iloc[-1])
            entry_price = entry_price if entry_price > 0 else current_close

            # 5. Compute Stop Loss
            atr = technical_data.get("atr", current_close * 0.02)
            swing_low = sr_data.get("recent_swing_low", current_close * 0.96)
            stop_loss = self._compute_stop_loss(entry_price, atr, swing_low, trade_type)

            # 6. Compute Targets (with minimum 1:2 R:R constraint)
            resistance_levels = sr_data.get("resistance_levels", [])
            fib_extensions = sr_data.get("fib_extensions", [])
            targets = self._compute_targets(entry_price, stop_loss, resistance_levels, fib_extensions)

            # Filter targets based on trade type:
            # INTRADAY: T1 only
            # SWING: T1 & T2
            # POSITIONAL: T1, T2 & T3
            if trade_type == "INTRADAY":
                targets = targets[:1]
            elif trade_type == "SWING":
                targets = targets[:2]
            else:
                targets = targets[:3]

            # 7. Compute Risk-Reward Ratio
            risk = abs(entry_price - stop_loss)
            primary_target = targets[0]["price"] if targets else entry_price + (2.0 * risk)
            rr_ratio = round((primary_target - entry_price) / risk, 2) if risk > 0 else 0.0

            # 8. Multi-signal Direction & Overall Confidence
            tech_bias = technical_data.get("bias", "NEUTRAL")  # "BULLISH", "BEARISH", "NEUTRAL"
            fund_bias = fundamental_data.get("bias", "NEUTRAL")
            ml_prob = ml_data.get("up_probability", 50.0)

            # Calculate composite confidence score (0-100%)
            tech_score = technical_data.get("confidence", 50.0)
            sr_score = entry_confidence
            fund_score = fundamental_data.get("confidence", 50.0)
            ml_score = ml_data.get("confidence", 50.0)

            overall_confidence = round(
                (0.35 * tech_score) + (0.25 * sr_score) + (0.20 * fund_score) + (0.20 * ml_score),
                1
            )

            # 9. Evaluate No-Trade Filter Rules
            no_trade_reasons = []

            # Rule A: Skip if India VIX > 25 (extreme volatility regime)
            if vix_val > 25.0:
                no_trade_reasons.append(
                    f"India VIX is elevated at {vix_val:.2f} (> 25.0), indicating an unpredictable high-volatility regime."
                )

            # Rule B: Skip if R:R < 1:2 (2.0)
            if rr_ratio < 2.0:
                no_trade_reasons.append(
                    f"Risk-Reward ratio is {rr_ratio:.2f}:1, which is below the mandatory minimum of 1:2.0."
                )

            # Rule C: Skip if overall confidence < 40%
            if overall_confidence < 40.0:
                no_trade_reasons.append(
                    f"Setup confidence score is {overall_confidence:.1f}%, which is below the minimum threshold of 40%."
                )

            # Rule D: Skip if conflicting signals (e.g. Bullish technicals but Bearish fundamentals)
            if tech_bias == "BULLISH" and fund_bias == "BEARISH":
                no_trade_reasons.append(
                    "Conflicting signals: Technical momentum is Bullish, but Fundamental valuation/health is Bearish."
                )
            elif tech_bias == "BEARISH" and fund_bias == "BULLISH":
                no_trade_reasons.append(
                    "Conflicting signals: Technical indicators are Bearish while Fundamentals remain Bullish."
                )

            # Determine Direction
            if no_trade_reasons:
                direction = "NO_TRADE"
            elif tech_bias == "BULLISH" and ml_prob >= 48.0:
                direction = "BUY"
            elif tech_bias == "BEARISH" and ml_prob < 45.0:
                direction = "SELL"
            else:
                direction = "NO_TRADE"
                no_trade_reasons.append("Indecisive market momentum; neither buyers nor sellers hold clear edge.")

            # 10. Position Sizing
            if direction != "NO_TRADE" and stop_loss < entry_price:
                position_size_data = self.sizer.calculate(
                    capital=capital,
                    risk_pct=risk_pct,
                    entry=entry_price,
                    stop_loss=stop_loss,
                    method="fixed_fractional",
                )
            else:
                position_size_data = {
                    "quantity": 0,
                    "capital_required": 0.0,
                    "capital_at_risk": 0.0,
                    "risk_per_share": round(risk, 2),
                    "risk_pct_of_capital": 0.0,
                    "confidence": 0.0,
                    "method": "fixed_fractional",
                }

            # 11. Trailing Stop Loss Rule (ATR-based trailing after T1)
            trailing_sl = {
                "activation_trigger": f"Target 1 reached (₹{targets[0]['price']:.2f})" if targets else "T1 Reached",
                "step_1": f"Move Stop-Loss to Breakeven (₹{entry_price:.2f}) immediately upon T1 fill.",
                "step_2": f"Trail by {1.5 if trade_type != 'POSITIONAL' else 2.0}x ATR (₹{atr * 1.5:.2f}) below subsequent swing highs.",
                "atr_multiplier": 1.5 if trade_type != "POSITIONAL" else 2.0,
                "atr_value": round(atr, 2),
            }

            # Compile Target fields
            t1_val = targets[0]["price"] if len(targets) > 0 else None
            t2_val = targets[1]["price"] if len(targets) > 1 else None
            t3_val = targets[2]["price"] if len(targets) > 2 else None

            # 12. Build Plan Data
            plan_data = {
                "symbol": clean_symbol,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "trade_type": trade_type,
                "direction": direction,
                "confidence": overall_confidence,
                "entry_price": round(entry_price, 2),
                "entry_zone": {
                    "min": round(min(entry_price * 0.995, entry_price - (0.2 * atr)), 2),
                    "max": round(max(entry_price * 1.005, entry_price + (0.2 * atr)), 2),
                    "confluence_score": entry_confidence,
                },
                "stop_loss": round(stop_loss, 2),
                "target_1": round(t1_val, 2) if t1_val else None,
                "target_2": round(t2_val, 2) if t2_val else None,
                "target_3": round(t3_val, 2) if t3_val else None,
                "targets": targets,
                "risk_reward": rr_ratio,
                "trailing_stop_loss": trailing_sl,
                "position_sizing": position_size_data,
                "position_size": position_size_data.get("quantity", 0),
                "capital_required": position_size_data.get("capital_required", 0.0),
                "indicators": {
                    "current_close": round(current_close, 2),
                    "rsi": round(technical_data.get("rsi", 50.0), 2),
                    "ema_20": round(technical_data.get("ema_20", current_close), 2),
                    "ema_50": round(technical_data.get("ema_50", current_close), 2),
                    "ema_200": round(technical_data.get("ema_200", current_close), 2),
                    "atr": round(atr, 2),
                    "volume_ratio": round(technical_data.get("volume_ratio", 1.0), 2),
                    "india_vix": round(vix_val, 2),
                },
                "support_levels": [round(s, 2) for s in sr_data.get("support_levels", [])[:3]],
                "resistance_levels": [round(r, 2) for r in sr_data.get("resistance_levels", [])[:3]],
                "fibonacci_levels": sr_data.get("fib_levels", {}),
                "no_trade_reasons": no_trade_reasons,
                "sebi_disclaimer": settings.disclaimer,
                "disclaimer": settings.disclaimer,
            }

            # 13. Generate Plain English Summary
            plan_data["summary"] = self._generate_summary(plan_data)

            return plan_data

        except Exception as e:
            logger.error(f"Error generating trade plan for {clean_symbol}: {e}", exc_info=True)
            return self._create_no_trade_plan(
                symbol=clean_symbol,
                trade_type=trade_type,
                reasons=[f"Exception occurred while analyzing {clean_symbol}: {str(e)}"],
                confidence=0.0,
            )

    def generate_all_types(self, symbol: str) -> Dict[str, Any]:
        """
        Generate trade plans across all three operational horizons:
        Intraday, Swing, and Positional.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary containing intraday, swing, and positional trade plans.
        """
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
        intraday = self.generate(clean_symbol, trade_type="INTRADAY")
        swing = self.generate(clean_symbol, trade_type="SWING")
        positional = self.generate(clean_symbol, trade_type="POSITIONAL")

        return {
            "symbol": clean_symbol,
            "intraday": intraday,
            "swing": swing,
            "positional": positional,
            "disclaimer": settings.disclaimer,
            "sebi_disclaimer": settings.disclaimer,
        }

    def _compute_entry_zone(
        self,
        technical_data: Dict[str, Any],
        sr_data: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Compute entry price and confluence confidence based on:
        1. Support level proximity
        2. Key EMA alignment (20/50/200)
        3. Fibonacci retracement levels (38.2%, 50%, 61.8%)
        4. Volume confirmation (above 20-day average)

        Args:
            technical_data: Calculated indicators
            sr_data: Support, resistance, and Fibonacci levels

        Returns:
            Tuple of (entry_price: float, confluence_confidence: float)
        """
        current_price = technical_data.get("current_close", 0.0)
        if current_price <= 0:
            return (0.0, 0.0)

        confluence_points: List[float] = []
        score = 0.0

        # Criterion 1: Near Support level (+25% confidence)
        supports = sr_data.get("support_levels", [])
        nearest_support = None
        for s in sorted(supports, reverse=True):
            if s <= current_price * 1.02:
                nearest_support = s
                break
        if nearest_support:
            dist_pct = abs(current_price - nearest_support) / current_price
            if dist_pct <= 0.03:  # within 3% of support
                confluence_points.append(nearest_support)
                score += 25.0

        # Criterion 2: Near Key EMA (+25% confidence)
        ema_20 = technical_data.get("ema_20", current_price)
        ema_50 = technical_data.get("ema_50", current_price)
        key_emas = [ema for ema in [ema_20, ema_50] if ema > 0]
        for ema in key_emas:
            dist_pct = abs(current_price - ema) / current_price
            if dist_pct <= 0.02:  # within 2% of EMA
                confluence_points.append(ema)
                score += 25.0
                break

        # Criterion 3: Near Fibonacci Retracement Level (+25% confidence)
        fib_levels = sr_data.get("fib_levels", {})
        for fib_key in ["0.382", "0.500", "0.618"]:
            if fib_key in fib_levels:
                f_val = fib_levels[fib_key]
                if abs(current_price - f_val) / current_price <= 0.025:
                    confluence_points.append(f_val)
                    score += 25.0
                    break

        # Criterion 4: Volume Confirmation (+25% confidence)
        volume_ratio = technical_data.get("volume_ratio", 1.0)
        if volume_ratio >= 1.2:
            score += 25.0
        elif volume_ratio >= 1.0:
            score += 15.0

        # Entry price calculation: weighted confluence or proximity
        if confluence_points:
            # Weighted average of confluence levels and current price
            entry_price = float(np.mean(confluence_points) * 0.4 + current_price * 0.6)
        else:
            entry_price = current_price

        # Minimum baseline confidence
        confidence = max(20.0, min(100.0, score))
        return (round(entry_price, 2), round(confidence, 1))

    def _compute_stop_loss(
        self,
        entry: float,
        atr: float,
        swing_low: float,
        trade_type: str = "SWING",
    ) -> float:
        """
        Compute optimal stop-loss below swing low or based on 1.5-2x ATR.
        Selects the logical tighter stop without being prone to market noise.

        Args:
            entry: Entry price
            atr: 14-period Average True Range
            swing_low: Recent swing low price
            trade_type: INTRADAY, SWING, or POSITIONAL

        Returns:
            Stop-loss price in INR
        """
        if entry <= 0:
            return 0.0

        atr = max(atr, entry * 0.01)

        # Multiplier according to operational timeframe
        if trade_type == "INTRADAY":
            atr_multiplier = 1.3
        elif trade_type == "SWING":
            atr_multiplier = 1.7
        else:  # POSITIONAL
            atr_multiplier = 2.2

        atr_stop = entry - (atr_multiplier * atr)

        # Swing low stop placed slightly below the low (0.5% buffer)
        swing_stop = swing_low * 0.995 if swing_low > 0 else atr_stop

        # Validation: Swing stop must be strictly below entry
        if swing_stop >= entry:
            swing_stop = entry - (1.5 * atr)

        # Logical selection: tighter but not too tight (at least 1.0x ATR buffer)
        min_allowed_stop = entry - (3.5 * atr)  # Don't risk too much
        max_allowed_stop = entry - (0.8 * atr)  # Don't place too close to avoid noise

        # Pick whichever is tighter (higher price) while respecting min noise buffer
        if min_allowed_stop <= swing_stop <= max_allowed_stop:
            # Swing stop is valid and well-placed
            chosen_sl = max(swing_stop, atr_stop)
        else:
            chosen_sl = atr_stop

        # Ensure chosen stop is strictly below entry
        if chosen_sl >= entry:
            chosen_sl = entry - (1.5 * atr)

        return round(chosen_sl, 2)

    def _compute_targets(
        self,
        entry: float,
        sl: float,
        resistance_levels: List[float],
        fib_extensions: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Compute multi-stage profit targets maintaining minimum 1:2 Risk-Reward ratio.

        T1: Next resistance / first Fibonacci extension (partial booking: 50%)
        T2: Second resistance / 127.2% Fibonacci extension
        T3: Major resistance / 161.8% Fibonacci extension

        Args:
            entry: Entry price
            sl: Stop loss price
            resistance_levels: Historical resistance prices
            fib_extensions: Fibonacci extension levels

        Returns:
            List of target dictionaries with target number, price, R:R, and booking %.
        """
        risk = max(entry - sl, entry * 0.01)
        min_rr = 2.0
        min_t1 = entry + (min_rr * risk)

        # Clean and sort resistances above min_t1
        valid_resistances = sorted([r for r in resistance_levels if r >= min_t1])

        # T1 candidate: First resistance >= min_t1, or 100% Fib / 2.0x risk
        if valid_resistances:
            t1 = valid_resistances[0]
        elif fib_extensions and fib_extensions[0] >= min_t1:
            t1 = fib_extensions[0]
        else:
            t1 = entry + (2.0 * risk)

        # T2 candidate: Second resistance >= T1 * 1.02 or 127.2% Fib or 3.0x risk
        min_t2 = max(t1 * 1.015, entry + (3.0 * risk))
        t2_candidates = [r for r in valid_resistances if r >= min_t2]
        if t2_candidates:
            t2 = t2_candidates[0]
        elif len(fib_extensions) > 1 and fib_extensions[1] >= min_t2:
            t2 = fib_extensions[1]
        else:
            t2 = entry + (3.2 * risk)

        # T3 candidate: Major resistance >= T2 * 1.02 or 161.8% Fib or 4.5x risk
        min_t3 = max(t2 * 1.02, entry + (4.5 * risk))
        t3_candidates = [r for r in valid_resistances if r >= min_t3]
        if t3_candidates:
            t3 = t3_candidates[0]
        elif len(fib_extensions) > 2 and fib_extensions[2] >= min_t3:
            t3 = fib_extensions[2]
        else:
            t3 = entry + (5.0 * risk)

        targets = [
            {
                "target": 1,
                "price": round(t1, 2),
                "risk_reward": round((t1 - entry) / risk, 2),
                "booking_percentage": 50.0,
                "description": "Next resistance / Fib extension (partial profit booking 50%)",
            },
            {
                "target": 2,
                "price": round(t2, 2),
                "risk_reward": round((t2 - entry) / risk, 2),
                "booking_percentage": 30.0,
                "description": "Second resistance / 127.2% Fib extension",
            },
            {
                "target": 3,
                "price": round(t3, 2),
                "risk_reward": round((t3 - entry) / risk, 2),
                "booking_percentage": 20.0,
                "description": "Major structural resistance / 161.8% Fib extension",
            },
        ]

        return targets

    def _generate_summary(self, plan_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable, institutional-grade summary in plain English.

        Example:
        "Bullish. Price above 20 & 50 EMA, RSI showing momentum at 62.
        Key support at ₹2,400 confirmed by Fibonacci 50%. Entry near ₹2,420,
        SL at ₹2,380 (1.5x ATR), Targets ₹2,500/₹2,550/₹2,620. R:R = 1:2.0.
        Risk: FII outflows and global uncertainty could invalidate this setup."

        Args:
            plan_data: Complete trade plan dictionary

        Returns:
            Plain English descriptive text.
        """
        direction = plan_data.get("direction", "NO_TRADE")
        symbol = plan_data.get("symbol", "Stock")
        ind = plan_data.get("indicators", {})
        rsi = ind.get("rsi", 50.0)
        close = ind.get("current_close", 0.0)
        ema_20 = ind.get("ema_20", close)
        ema_50 = ind.get("ema_50", close)
        atr = ind.get("atr", close * 0.02)
        vix = ind.get("india_vix", 15.0)

        entry = plan_data.get("entry_price", close)
        sl = plan_data.get("stop_loss", 0.0)
        targets = plan_data.get("targets", [])
        rr = plan_data.get("risk_reward", 2.0)
        reasons = plan_data.get("no_trade_reasons", [])

        if direction == "NO_TRADE":
            reasons_str = " ".join(reasons) if reasons else "Confluence criteria not satisfied."
            return (
                f"NO_TRADE on {symbol}. Setup does not satisfy strict execution criteria: {reasons_str} "
                f"Current Price: ₹{close:.2f}, RSI: {rsi:.1f}, India VIX: {vix:.2f}. "
                f"Capital preservation is prioritized until high-probability confluence emerges."
            )

        # Trend and EMA commentary
        if close > ema_20 and close > ema_50:
            trend_str = "Price trading firmly above 20 and 50 EMA indicating strong bullish trend alignment"
        elif close > ema_50:
            trend_str = "Price consolidating above 50 EMA with positive underlying breadth"
        else:
            trend_str = "Price testing key dynamic moving averages"

        # RSI momentum commentary
        if rsi >= 60:
            rsi_str = f"RSI showing healthy bullish momentum at {rsi:.1f}"
        elif rsi >= 45:
            rsi_str = f"RSI in constructive accumulation territory at {rsi:.1f}"
        else:
            rsi_str = f"RSI stabilizing from oversold levels at {rsi:.1f}"

        # Target string
        t_strings = [f"₹{t['price']:.2f}" for t in targets]
        target_display = "/".join(t_strings) if t_strings else f"₹{entry * 1.05:.2f}"

        # ATR multiple representation
        atr_mult = round(abs(entry - sl) / atr, 1) if atr > 0 else 1.5

        # Key support reference
        supports = plan_data.get("support_levels", [])
        sup_str = f"Key support located at ₹{supports[0]:.2f}" if supports else "Support validated by recent price action"

        summary = (
            f"{'Bullish' if direction == 'BUY' else 'Bearish'} setup on {symbol}. "
            f"{trend_str}, with {rsi_str}. {sup_str}. "
            f"Entry near ₹{entry:.2f}, SL at ₹{sl:.2f} ({atr_mult}x ATR), Targets {target_display}. "
            f"R:R = 1:{rr:.1f}. "
            f"Risk: Macro headline volatility, India VIX fluctuations ({vix:.2f}), or FII institutional flow shifts "
            f"could invalidate this setup."
        )

        return summary

    def _fetch_market_data(self, symbol: str, trade_type: str) -> Dict[str, Any]:
        """Fetch OHLCV historical dataframe for the required horizon."""
        if trade_type == "INTRADAY":
            # 15-minute or short-period daily
            df = self.fetcher.get_ohlcv(symbol, period="1mo", interval="1d")
        elif trade_type == "SWING":
            df = self.fetcher.get_ohlcv(symbol, period="1y", interval="1d")
        else:  # POSITIONAL
            df = self.fetcher.get_ohlcv(symbol, period="2y", interval="1d")

        return {"df": df}

    def _analyze_technicals(self, symbol: str, df: pd.DataFrame, trade_type: str) -> Dict[str, Any]:
        """Compute key technical indicators using pandas-ta with robust fallback."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Exponential Moving Averages
        if ta is not None:
            try:
                ema_20 = float(ta.ema(close, length=20).iloc[-1])
                ema_50 = float(ta.ema(close, length=50).iloc[-1])
                ema_200 = float(ta.ema(close, length=min(200, len(close) - 1)).iloc[-1])
                rsi = float(ta.rsi(close, length=14).iloc[-1])
                atr_s = ta.atr(high, low, close, length=14)
                atr = float(atr_s.iloc[-1]) if atr_s is not None else float((high - low).mean())
            except Exception:
                ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
                ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                ema_200 = float(close.ewm(span=min(200, len(close)), adjust=False).mean().iloc[-1])
                rsi = self._calc_rsi(close)
                atr = self._calc_atr(high, low, close)
        else:
            ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            ema_200 = float(close.ewm(span=min(200, len(close)), adjust=False).mean().iloc[-1])
            rsi = self._calc_rsi(close)
            atr = self._calc_atr(high, low, close)

        current_close = float(close.iloc[-1])
        vol_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        volume_ratio = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

        # Technical bias and confidence calculation
        bullish_score = 0
        if current_close > ema_20:
            bullish_score += 25
        if current_close > ema_50:
            bullish_score += 25
        if 48 <= rsi <= 72:
            bullish_score += 30
        elif rsi > 72:
            bullish_score += 10  # Overbought penalty
        if volume_ratio >= 1.0:
            bullish_score += 20

        bias = "BULLISH" if bullish_score >= 60 else ("BEARISH" if bullish_score <= 35 else "NEUTRAL")

        return {
            "current_close": current_close,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": ema_200,
            "rsi": rsi,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "bias": bias,
            "confidence": float(bullish_score),
        }

    def _analyze_support_resistance(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute support, resistance, swing points, and Fibonacci retracement/extensions."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        recent_high = float(high.iloc[-40:].max()) if len(high) >= 40 else float(high.max())
        recent_low = float(low.iloc[-40:].min()) if len(low) >= 40 else float(low.min())
        diff = recent_high - recent_low if recent_high > recent_low else 1.0

        # Fibonacci Retracements (from recent swing low to swing high)
        fib_levels = {
            "0.236": round(recent_high - (0.236 * diff), 2),
            "0.382": round(recent_high - (0.382 * diff), 2),
            "0.500": round(recent_high - (0.500 * diff), 2),
            "0.618": round(recent_high - (0.618 * diff), 2),
            "0.786": round(recent_high - (0.786 * diff), 2),
        }

        # Fibonacci Extensions
        fib_extensions = [
            round(recent_high + (0.618 * diff), 2),
            round(recent_high + (1.000 * diff), 2),
            round(recent_high + (1.272 * diff), 2),
            round(recent_high + (1.618 * diff), 2),
        ]

        # Pivot Points based on recent bar
        h_last = float(high.iloc[-2]) if len(high) > 1 else float(high.iloc[-1])
        l_last = float(low.iloc[-2]) if len(low) > 1 else float(low.iloc[-1])
        c_last = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
        pivot = (h_last + l_last + c_last) / 3.0

        r1 = (2 * pivot) - l_last
        s1 = (2 * pivot) - h_last
        r2 = pivot + (h_last - l_last)
        s2 = pivot - (h_last - l_last)

        # Collect Support & Resistance levels
        supports = sorted(list({round(s1, 2), round(s2, 2), round(recent_low, 2), fib_levels["0.618"]}), reverse=True)
        resistances = sorted(list({round(r1, 2), round(r2, 2), round(recent_high, 2), fib_extensions[0]}))

        return {
            "support_levels": supports,
            "resistance_levels": resistances,
            "recent_swing_low": recent_low,
            "recent_swing_high": recent_high,
            "fib_levels": fib_levels,
            "fib_extensions": fib_extensions,
            "pivot": round(pivot, 2),
        }

    def _analyze_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data and derive fundamental health bias."""
        try:
            if FundamentalAnalyzer is not None:
                analyzer = FundamentalAnalyzer()
                if hasattr(analyzer, "analyze"):
                    return analyzer.analyze(symbol)

            fund_data = self.fetcher.get_fundamentals(symbol)
            if not fund_data:
                return {"bias": "NEUTRAL", "confidence": 50.0}

            pe = fund_data.get("pe_ratio")
            pb = fund_data.get("pb_ratio")
            roe = fund_data.get("roe")
            debt_to_equity = fund_data.get("debt_to_equity")

            score = 50.0
            if pe and 5 < pe < 35:
                score += 15.0
            if roe and roe > 0.15:
                score += 20.0
            if debt_to_equity and debt_to_equity < 1.0:
                score += 15.0

            bias = "BULLISH" if score >= 65.0 else ("BEARISH" if score <= 40.0 else "NEUTRAL")
            return {"bias": bias, "confidence": min(100.0, score), "details": fund_data}

        except Exception as e:
            logger.debug(f"Fundamental analysis fallback used for {symbol}: {e}")
            return {"bias": "NEUTRAL", "confidence": 50.0}

    def _analyze_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML directional probability score."""
        try:
            if PredictionEngine is not None:
                engine = PredictionEngine()
                if hasattr(engine, "predict"):
                    res = engine.predict(symbol)
                    return {
                        "up_probability": res.get("up_probability", 55.0),
                        "confidence": res.get("confidence", 60.0),
                    }

            # Quantitative logistic momentum model
            close = df["Close"]
            ret_5d = (close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
            ret_20d = (close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0

            z = (ret_5d * 10.0) + (ret_20d * 5.0)
            up_prob = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, z))))
            prob_pct = round(up_prob * 100.0, 1)

            return {
                "up_probability": prob_pct,
                "confidence": round(abs(prob_pct - 50.0) * 2.0, 1),
            }

        except Exception as e:
            logger.debug(f"ML prediction fallback used for {symbol}: {e}")
            return {"up_probability": 50.0, "confidence": 50.0}

    def _fetch_india_vix(self) -> Dict[str, Any]:
        """Fetch current India VIX index value."""
        try:
            vix_df = self.fetcher.get_india_vix(period="5d")
            if vix_df is not None and not vix_df.empty:
                val = float(vix_df["Close"].iloc[-1])
                return {"value": val, "status": "ok"}
        except Exception as e:
            logger.warning(f"Could not fetch India VIX: {e}")
        return {"value": 14.5, "status": "default"}

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Pure-pandas vectorized RSI calculation."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(max(0.0, min(100.0, rsi)))

    def _calc_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Pure-pandas vectorized ATR calculation."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
        return float(atr_val) if not np.isnan(atr_val) else float((high - low).mean())

    def _create_no_trade_plan(
        self,
        symbol: str,
        trade_type: str,
        reasons: List[str],
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """Build standardized NO_TRADE payload."""
        return {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "trade_type": trade_type,
            "direction": "NO_TRADE",
            "confidence": confidence,
            "entry_price": 0.0,
            "entry_zone": {"min": 0.0, "max": 0.0, "confluence_score": 0.0},
            "stop_loss": 0.0,
            "target_1": None,
            "target_2": None,
            "target_3": None,
            "targets": [],
            "risk_reward": 0.0,
            "trailing_stop_loss": {},
            "position_sizing": {"quantity": 0, "capital_required": 0.0, "capital_at_risk": 0.0},
            "indicators": {},
            "support_levels": [],
            "resistance_levels": [],
            "fibonacci_levels": {},
            "no_trade_reasons": reasons,
            "summary": f"NO_TRADE on {symbol}. Setup disqualified: {' '.join(reasons)}",
            "sebi_disclaimer": settings.disclaimer,
            "disclaimer": settings.disclaimer,
        }
