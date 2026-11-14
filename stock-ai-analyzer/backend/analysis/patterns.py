"""
Pattern recognition module for Indian stock market analysis (NSE/BSE).

Provides algorithmic detection for:
1. Candlestick patterns (Doji, Hammer, Engulfing, Morning/Evening Star, etc.)
   using pandas_ta and verified OHLC candle geometry.
2. Classical chart patterns (Head & Shoulders, Double Top/Bottom, Triangles,
   Flags, Cup & Handle, Breakouts, Fakeouts) using extrema detection on
   rolling windows (60-120 bars).

Historical win rates are based on Thomas Bulkowski's empirical statistics
from the 'Encyclopedia of Candlestick Charts' and 'Encyclopedia of Chart Patterns'.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from scipy.signal import argrelextrema
except ImportError:
    argrelextrema = None

logger = logging.getLogger(__name__)

# Mandatory SEBI Educational & Risk Disclaimer
SEBI_DISCLAIMER: str = (
    "⚠️ DISCLAIMER: This is an educational tool for stock market analysis. "
    "It is NOT SEBI-registered investment advice. Past performance does not "
    "guarantee future results. No prediction model is 100% accurate. "
    "Always do your own research and consult a SEBI-registered financial "
    "advisor before making investment decisions. Trade at your own risk."
)

# Generally accepted academic & empirical win rates (Bulkowski Statistics)
BULKOWSKI_WIN_RATES: Dict[str, float] = {
    # Candlestick Patterns (Reversal / Continuation rates)
    "Doji": 52.0,
    "Dragonfly Doji": 53.0,
    "Gravestone Doji": 52.0,
    "Hammer": 60.0,
    "Inverted Hammer": 60.0,
    "Bullish Engulfing": 63.0,
    "Bearish Engulfing": 79.0,
    "Morning Star": 78.0,
    "Evening Star": 72.0,
    "Three White Soldiers": 82.0,
    "Three Black Crows": 78.0,
    "Spinning Top": 51.0,
    "Bullish Marubozu": 56.0,
    "Bearish Marubozu": 56.0,
    "Bullish Harami": 53.0,
    "Bearish Harami": 53.0,
    "Piercing Pattern": 64.0,
    "Dark Cloud Cover": 60.0,
    "Hanging Man": 59.0,
    "Shooting Star": 59.0,
    # Chart Patterns (Breakout / Target achievement rates)
    "Head & Shoulders": 89.0,
    "Inverse Head & Shoulders": 89.0,
    "Double Top": 75.0,
    "Double Bottom": 78.0,
    "Ascending Triangle": 72.0,
    "Descending Triangle": 72.0,
    "Symmetrical Triangle": 63.0,
    "Bull Flag": 67.0,
    "Bear Flag": 67.0,
    "Cup & Handle": 79.0,
    "Breakout": 65.0,
    "Fakeout": 70.0,
}


class PatternRecognizer:
    """
    Production-grade Candlestick and Chart Pattern Recognizer for Stock AI Analyzer.

    Supports:
    - 20 Japanese Candlestick patterns with pandas_ta and geometric verification.
    - 12 Classical Chart patterns using local extrema detection on 60-120 bar windows.
    - Standardized output with confidence scores (0-100%), directional bias,
      and Bulkowski historical win rates.
    """

    SEBI_DISCLAIMER: str = SEBI_DISCLAIMER
    BULKOWSKI_WIN_RATES: Dict[str, float] = BULKOWSKI_WIN_RATES

    def __init__(self, default_window: int = 100):
        """
        Initialize PatternRecognizer.

        Args:
            default_window: Default rolling bar window for chart patterns (60-120).
        """
        self.default_window = max(60, min(120, default_window))

    # =========================================================================
    # Data Normalization & Mathematical Utilities
    # =========================================================================

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """
        Normalize stock symbol to NSE format with .NS suffix.

        Args:
            symbol: Stock ticker (e.g., 'RELIANCE', 'TCS.NS', 'infy').

        Returns:
            Normalized symbol string with '.NS' suffix for NSE.
        """
        symbol = symbol.strip().upper()
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
            return f"{symbol}.NS"
        return symbol

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate, clean, and standardize OHLCV dataframe columns.

        Args:
            df: Input price dataframe.

        Returns:
            Cleaned dataframe with lowercase 'open', 'high', 'low', 'close', 'volume'.

        Raises:
            ValueError: If dataframe is empty, invalid, or missing essential columns.
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input data must be a non-empty pandas DataFrame.")

        df_clean = df.copy()

        # Handle MultiIndex columns (e.g. from yfinance download)
        if isinstance(df_clean.columns, pd.MultiIndex):
            df_clean.columns = [c[0] if isinstance(c, tuple) else c for c in df_clean.columns]

        # Case-insensitive column map
        col_map = {str(c).strip(): str(c).strip().lower() for c in df_clean.columns}
        df_clean.rename(columns=col_map, inplace=True)

        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df_clean.columns]
        if missing:
            raise ValueError(f"DataFrame is missing essential OHLC columns: {missing}")

        for col in required:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if "volume" in df_clean.columns:
            df_clean["volume"] = pd.to_numeric(df_clean["volume"], errors="coerce").fillna(1.0)
        else:
            df_clean["volume"] = 1.0

        # Drop NaN or invalid rows
        df_clean.dropna(subset=required, inplace=True)
        df_clean = df_clean[(df_clean["high"] >= df_clean["low"]) & (df_clean["close"] > 0)]
        df_clean.reset_index(drop=True, inplace=True)

        if len(df_clean) < 3:
            raise ValueError("DataFrame has fewer than 3 valid price bars after cleaning.")

        return df_clean

    def _empty_result(
        self,
        pattern: str,
        direction: str = "NEUTRAL",
        historical_win_rate: Optional[float] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Generate standardized result dictionary when a pattern is NOT detected."""
        win_rate = (
            historical_win_rate
            if historical_win_rate is not None
            else BULKOWSKI_WIN_RATES.get(pattern, 50.0)
        )
        return {
            "detected": False,
            "pattern": pattern,
            "direction": direction,
            "confidence": 0.0,
            "historical_win_rate": float(win_rate),
            "description": description or f"No {pattern} pattern detected.",
        }

    def _find_peaks_and_troughs(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        order: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify local extrema (peaks and troughs) using scipy or rolling window.

        Args:
            highs: Array of high prices.
            lows: Array of low prices.
            order: Number of points on each side to compare.

        Returns:
            Tuple of (peak_indices, trough_indices).
        """
        n = len(highs)
        if n < order * 2 + 1:
            order = max(2, n // 8)
            if n < order * 2 + 1:
                return np.array([], dtype=int), np.array([], dtype=int)

        if argrelextrema is not None:
            try:
                peaks = argrelextrema(highs, np.greater_equal, order=order)[0]
                troughs = argrelextrema(lows, np.less_equal, order=order)[0]
                return peaks, troughs
            except Exception as e:
                logger.debug(f"scipy argrelextrema error: {e}, using rolling extrema fallback.")

        # Robust pure-numpy rolling extrema fallback
        peaks = []
        troughs = []
        for i in range(order, n - order):
            if np.all(highs[i] >= highs[i - order : i]) and np.all(highs[i] > highs[i + 1 : i + order + 1]):
                peaks.append(i)
            if np.all(lows[i] <= lows[i - order : i]) and np.all(lows[i] < lows[i + 1 : i + order + 1]):
                troughs.append(i)

        return np.array(peaks, dtype=int), np.array(troughs, dtype=int)

    def _fit_trendline(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Fit a linear regression trendline to (x, y) coordinates.

        Returns:
            Tuple of (slope, intercept, r_squared).
        """
        if len(x) < 2 or len(x) != len(y):
            return 0.0, 0.0, 0.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        dx = x - x_mean
        dy = y - y_mean
        sum_dx_sq = np.sum(dx**2)

        if sum_dx_sq < 1e-12:
            return 0.0, float(y_mean), 0.0

        slope = float(np.sum(dx * dy) / sum_dx_sq)
        intercept = float(y_mean - slope * x_mean)

        y_pred = slope * x + intercept
        ss_tot = np.sum(dy**2)
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared = float(1.0 - (ss_res / max(ss_tot, 1e-12))) if ss_tot > 1e-12 else 0.0

        return slope, intercept, max(0.0, min(1.0, r_squared))

    def _detect_prior_trend(self, df: pd.DataFrame, bars_back: int = 5) -> str:
        """
        Determine short-term trend prior to the pattern formation.

        Returns:
            'UP', 'DOWN', or 'SIDEWAYS'.
        """
        if len(df) < bars_back + 2:
            return "SIDEWAYS"

        recent_close = float(df["close"].iloc[-2])
        prior_close = float(df["close"].iloc[-bars_back - 1])
        change_pct = (recent_close - prior_close) / max(prior_close, 1e-6)

        if change_pct >= 0.012:
            return "UP"
        elif change_pct <= -0.012:
            return "DOWN"
        return "SIDEWAYS"

    # =========================================================================
    # SECTION 1: CANDLESTICK PATTERNS
    # =========================================================================

    def detect_doji(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Regular Doji pattern.

        Open and close are virtually identical (body <= 5% of range) with balanced
        upper and lower shadows, indicating market hesitation.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            # Try pandas_ta CDL function
            ta_detected = False
            try:
                if hasattr(ta, "cdl_doji"):
                    res = ta.cdl_doji(
                        open_=clean_df["open"],
                        high=clean_df["high"],
                        low=clean_df["low"],
                        close=clean_df["close"],
                    )
                    if res is not None and not res.empty and res.iloc[-1] != 0:
                        ta_detected = True
            except Exception as e:
                logger.debug(f"pandas_ta cdl_doji check: {e}")

            # Doji geometric rules
            is_doji_body = (body / rng) <= 0.05
            has_upper_wick = (upper_shadow / rng) >= 0.20
            has_lower_wick = (lower_shadow / rng) >= 0.20
            is_regular = is_doji_body and has_upper_wick and has_lower_wick

            if is_regular or ta_detected:
                base_conf = 55.0
                if body / rng <= 0.02:
                    base_conf += 15.0
                if abs(upper_shadow - lower_shadow) / rng <= 0.15:
                    base_conf += 15.0
                if len(clean_df) >= 20:
                    vol_sma = clean_df["volume"].iloc[-20:].mean()
                    if float(c["volume"]) > vol_sma:
                        base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Doji",
                    "direction": "NEUTRAL",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Doji"],
                    "description": (
                        f"Regular Doji detected at ₹{cl:.2f}: Open (₹{o:.2f}) and Close (₹{cl:.2f}) "
                        f"are virtually equal (body {body/rng*100:.1f}% of range), signaling indecision."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_doji: {e}")

        return self._empty_result("Doji", "NEUTRAL")

    def detect_dragonfly_doji(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Dragonfly Doji pattern.

        Open and close are near the high (minimal upper shadow), with a very long
        lower shadow representing aggressive rejection of lower prices by bulls.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            is_dragonfly = (
                (body / rng <= 0.05)
                and (upper_shadow / rng <= 0.08)
                and (lower_shadow / rng >= 0.65)
            )

            if is_dragonfly:
                prior_trend = self._detect_prior_trend(clean_df)
                base_conf = 58.0
                if lower_shadow / rng >= 0.75:
                    base_conf += 15.0
                if prior_trend == "DOWN":
                    base_conf += 15.0  # Classic bullish bottom reversal
                if body / rng <= 0.02:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Dragonfly Doji",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Dragonfly Doji"],
                    "description": (
                        f"Dragonfly Doji detected at ₹{cl:.2f}: Long lower shadow ({lower_shadow/rng*100:.1f}%) "
                        f"shows strong buying rejection off session low of ₹{l:.2f}."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_dragonfly_doji: {e}")

        return self._empty_result("Dragonfly Doji", "BULLISH")

    def detect_gravestone_doji(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Gravestone Doji pattern.

        Open and close are near the low (minimal lower shadow), with a very long
        upper shadow representing heavy rejection of higher prices by bears.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            is_gravestone = (
                (body / rng <= 0.05)
                and (lower_shadow / rng <= 0.08)
                and (upper_shadow / rng >= 0.65)
            )

            if is_gravestone:
                prior_trend = self._detect_prior_trend(clean_df)
                base_conf = 58.0
                if upper_shadow / rng >= 0.75:
                    base_conf += 15.0
                if prior_trend == "UP":
                    base_conf += 15.0  # Classic bearish top reversal
                if body / rng <= 0.02:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Gravestone Doji",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Gravestone Doji"],
                    "description": (
                        f"Gravestone Doji detected at ₹{cl:.2f}: Long upper shadow ({upper_shadow/rng*100:.1f}%) "
                        f"signals strong rejection of session high of ₹{h:.2f}."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_gravestone_doji: {e}")

        return self._empty_result("Gravestone Doji", "BEARISH")

    def detect_hammer(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Hammer pattern.

        Small real body near the top of the range with a long lower shadow (>= 2x body)
        and minimal upper shadow, appearing after a downward move.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l
            prior_trend = self._detect_prior_trend(clean_df)

            is_hammer_shape = (
                (body / rng <= 0.35)
                and (lower_shadow >= 2.0 * body)
                and (lower_shadow / rng >= 0.55)
                and (upper_shadow / rng <= 0.12)
            )

            # Hammer specifically requires a prior decline or neutral-down context
            if is_hammer_shape and prior_trend != "UP":
                base_conf = 60.0
                if lower_shadow >= 3.0 * body:
                    base_conf += 10.0
                if cl >= o:  # Green/bullish hammer
                    base_conf += 10.0
                if prior_trend == "DOWN":
                    base_conf += 10.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean():
                    base_conf += 8.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Hammer",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Hammer"],
                    "description": (
                        f"Hammer detected at ₹{cl:.2f}: Small body with lower shadow {lower_shadow/max(body, 1e-6):.1f}x "
                        f"body height following a downward move, indicating strong demand."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_hammer: {e}")

        return self._empty_result("Hammer", "BULLISH")

    def detect_inverted_hammer(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Inverted Hammer pattern.

        Small real body near the bottom with a long upper shadow (>= 2x body)
        and little/no lower shadow, appearing after a downtrend.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l
            prior_trend = self._detect_prior_trend(clean_df)

            is_inv_hammer = (
                (body / rng <= 0.35)
                and (upper_shadow >= 2.0 * body)
                and (upper_shadow / rng >= 0.55)
                and (lower_shadow / rng <= 0.12)
            )

            if is_inv_hammer and prior_trend != "UP":
                base_conf = 58.0
                if upper_shadow >= 3.0 * body:
                    base_conf += 10.0
                if cl >= o:
                    base_conf += 8.0
                if prior_trend == "DOWN":
                    base_conf += 12.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean():
                    base_conf += 8.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Inverted Hammer",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Inverted Hammer"],
                    "description": (
                        f"Inverted Hammer detected at ₹{cl:.2f}: Long upper shadow {upper_shadow/max(body, 1e-6):.1f}x "
                        f"body height indicates bulls testing higher prices during downtrend."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_inverted_hammer: {e}")

        return self._empty_result("Inverted Hammer", "BULLISH")

    def detect_hanging_man(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Hanging Man pattern.

        Structurally identical to a hammer (small body at top, long lower shadow >= 2x body),
        but occurs at the crest of an UPTREND, signaling potential bearish exhaustion.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l
            prior_trend = self._detect_prior_trend(clean_df)

            is_hanging_shape = (
                (body / rng <= 0.35)
                and (lower_shadow >= 2.0 * body)
                and (lower_shadow / rng >= 0.55)
                and (upper_shadow / rng <= 0.12)
            )

            if is_hanging_shape and prior_trend == "UP":
                base_conf = 60.0
                if cl < o:  # Red/bearish body is more significant
                    base_conf += 12.0
                if lower_shadow >= 3.0 * body:
                    base_conf += 10.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean():
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Hanging Man",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Hanging Man"],
                    "description": (
                        f"Hanging Man detected at ₹{cl:.2f}: Long lower shadow after an uptrend warns "
                        f"that aggressive selling pressure has breached bullish control."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_hanging_man: {e}")

        return self._empty_result("Hanging Man", "BEARISH")

    def detect_shooting_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Shooting Star pattern.

        Small real body near the bottom of the range with a long upper shadow (>= 2x body),
        appearing at the peak of an UPTREND, signaling bearish reversal.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l
            prior_trend = self._detect_prior_trend(clean_df)

            is_star_shape = (
                (body / rng <= 0.35)
                and (upper_shadow >= 2.0 * body)
                and (upper_shadow / rng >= 0.55)
                and (lower_shadow / rng <= 0.12)
            )

            if is_star_shape and prior_trend == "UP":
                base_conf = 62.0
                if cl < o:  # Red body confirms bears took over
                    base_conf += 12.0
                if upper_shadow >= 3.0 * body:
                    base_conf += 10.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean():
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Shooting Star",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Shooting Star"],
                    "description": (
                        f"Shooting Star detected at ₹{cl:.2f}: High at ₹{h:.2f} was aggressively rejected, "
                        f"leaving a long upper shadow ({upper_shadow/rng*100:.1f}%) at the top of an advance."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_shooting_star: {e}")

        return self._empty_result("Shooting Star", "BEARISH")

    def detect_bullish_engulfing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bullish Engulfing pattern.

        2-bar pattern: Bar 1 is bearish; Bar 2 is a strong bullish candle whose real body
        completely engulfs the real body of Bar 1.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Bullish Engulfing", "BULLISH")

            c1 = clean_df.iloc[-2]
            c2 = clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            body1 = abs(cl1 - o1)
            body2 = abs(cl2 - o2)

            is_bear1 = cl1 < o1
            is_bull2 = cl2 > o2
            engulfs_body = (o2 <= cl1) and (cl2 >= o1) and (body2 > body1)

            if is_bear1 and is_bull2 and engulfs_body:
                base_conf = 65.0
                if body2 >= 1.5 * body1:
                    base_conf += 12.0
                if (h2 >= h1) and (l2 <= l1):  # Full range engulfing
                    base_conf += 10.0
                if float(c2["volume"]) > float(c1["volume"]) * 1.25:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bullish Engulfing",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bullish Engulfing"],
                    "description": (
                        f"Bullish Engulfing detected: Green candle (₹{o2:.2f} to ₹{cl2:.2f}) completely covers "
                        f"previous red candle (₹{o1:.2f} to ₹{cl1:.2f}), indicating strong buyer takeover."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_bullish_engulfing: {e}")

        return self._empty_result("Bullish Engulfing", "BULLISH")

    def detect_bearish_engulfing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bearish Engulfing pattern.

        2-bar pattern: Bar 1 is bullish; Bar 2 is a strong bearish candle whose real body
        completely engulfs the real body of Bar 1.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Bearish Engulfing", "BEARISH")

            c1 = clean_df.iloc[-2]
            c2 = clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            body1 = abs(cl1 - o1)
            body2 = abs(cl2 - o2)

            is_bull1 = cl1 > o1
            is_bear2 = cl2 < o2
            engulfs_body = (o2 >= cl1) and (cl2 <= o1) and (body2 > body1)

            if is_bull1 and is_bear2 and engulfs_body:
                base_conf = 68.0
                if body2 >= 1.5 * body1:
                    base_conf += 12.0
                if (h2 >= h1) and (l2 <= l1):
                    base_conf += 10.0
                if float(c2["volume"]) > float(c1["volume"]) * 1.25:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bearish Engulfing",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bearish Engulfing"],
                    "description": (
                        f"Bearish Engulfing detected: Red candle (₹{o2:.2f} to ₹{cl2:.2f}) completely covers "
                        f"previous green candle (₹{o1:.2f} to ₹{cl1:.2f}), showing heavy institutional selling."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_bearish_engulfing: {e}")

        return self._empty_result("Bearish Engulfing", "BEARISH")

    def detect_morning_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Morning Star pattern.

        3-bar bullish reversal pattern:
        1. Long bearish candle.
        2. Small real body (star) gapping or sitting below Bar 1.
        3. Strong bullish candle closing well into Bar 1's body (> 50% midpoint).
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 3:
                return self._empty_result("Morning Star", "BULLISH")

            c1, c2, c3 = clean_df.iloc[-3], clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])
            o3, h3, l3, cl3 = float(c3["open"]), float(c3["high"]), float(c3["low"]), float(c3["close"])

            rng1 = max(h1 - l1, 1e-6)
            body1 = abs(cl1 - o1)
            rng2 = max(h2 - l2, 1e-6)
            body2 = abs(cl2 - o2)

            is_bear1 = (cl1 < o1) and (body1 / rng1 >= 0.45)
            is_star2 = body2 / rng2 <= 0.35 and max(o2, cl2) <= max(o1, cl1)
            midpoint1 = (o1 + cl1) / 2.0
            is_bull3 = (cl3 > o3) and (cl3 >= midpoint1)

            if is_bear1 and is_star2 and is_bull3:
                base_conf = 72.0
                if cl3 >= (o1 * 0.7 + cl1 * 0.3):  # Deep penetration into Bar 1
                    base_conf += 10.0
                if min(o2, cl2) < cl1:  # Gap down on star
                    base_conf += 8.0
                if float(c3["volume"]) > float(c2["volume"]):
                    base_conf += 8.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Morning Star",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Morning Star"],
                    "description": (
                        f"Morning Star detected: 3-bar bottom reversal with star at ₹{min(o2,cl2):.2f} "
                        f"and third green bar closing at ₹{cl3:.2f} (above midpoint ₹{midpoint1:.2f})."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_morning_star: {e}")

        return self._empty_result("Morning Star", "BULLISH")

    def detect_evening_star(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Evening Star pattern.

        3-bar bearish reversal pattern:
        1. Long bullish candle.
        2. Small real body (star) sitting above Bar 1.
        3. Strong bearish candle closing well into Bar 1's body (< 50% midpoint).
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 3:
                return self._empty_result("Evening Star", "BEARISH")

            c1, c2, c3 = clean_df.iloc[-3], clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])
            o3, h3, l3, cl3 = float(c3["open"]), float(c3["high"]), float(c3["low"]), float(c3["close"])

            rng1 = max(h1 - l1, 1e-6)
            body1 = abs(cl1 - o1)
            rng2 = max(h2 - l2, 1e-6)
            body2 = abs(cl2 - o2)

            is_bull1 = (cl1 > o1) and (body1 / rng1 >= 0.45)
            is_star2 = body2 / rng2 <= 0.35 and min(o2, cl2) >= min(o1, cl1)
            midpoint1 = (o1 + cl1) / 2.0
            is_bear3 = (cl3 < o3) and (cl3 <= midpoint1)

            if is_bull1 and is_star2 and is_bear3:
                base_conf = 72.0
                if cl3 <= (o1 * 0.3 + cl1 * 0.7):  # Deep downward penetration
                    base_conf += 10.0
                if max(o2, cl2) > cl1:
                    base_conf += 8.0
                if float(c3["volume"]) > float(c2["volume"]):
                    base_conf += 8.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Evening Star",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Evening Star"],
                    "description": (
                        f"Evening Star detected: 3-bar top reversal with star at ₹{max(o2,cl2):.2f} "
                        f"and third red bar closing at ₹{cl3:.2f} (below midpoint ₹{midpoint1:.2f})."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_evening_star: {e}")

        return self._empty_result("Evening Star", "BEARISH")

    def detect_three_white_soldiers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Three White Soldiers pattern.

        3 consecutive long bullish candles, each closing higher than the previous,
        opening within the prior candle body, with relatively short upper shadows.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 3:
                return self._empty_result("Three White Soldiers", "BULLISH")

            bars = [clean_df.iloc[-3], clean_df.iloc[-2], clean_df.iloc[-1]]
            opens = [float(b["open"]) for b in bars]
            highs = [float(b["high"]) for b in bars]
            lows = [float(b["low"]) for b in bars]
            closes = [float(b["close"]) for b in bars]

            all_bullish = all(closes[i] > opens[i] for i in range(3))
            ascending_closes = (closes[2] > closes[1] > closes[0])
            valid_opens = (
                (opens[1] >= opens[0] and opens[1] <= closes[0])
                and (opens[2] >= opens[1] and opens[2] <= closes[1])
            )
            short_upper_wicks = all(
                (highs[i] - closes[i]) / max(highs[i] - lows[i], 1e-6) <= 0.25 for i in range(3)
            )

            if all_bullish and ascending_closes and valid_opens and short_upper_wicks:
                base_conf = 75.0
                if all((highs[i] - closes[i]) / max(highs[i] - lows[i], 1e-6) <= 0.15 for i in range(3)):
                    base_conf += 10.0
                vols = [float(b["volume"]) for b in bars]
                if vols[2] >= vols[1] >= vols[0]:
                    base_conf += 12.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Three White Soldiers",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Three White Soldiers"],
                    "description": (
                        f"Three White Soldiers detected: Three consecutive strong green candles "
                        f"closed at ₹{closes[0]:.2f}, ₹{closes[1]:.2f}, and ₹{closes[2]:.2f} near their highs."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_three_white_soldiers: {e}")

        return self._empty_result("Three White Soldiers", "BULLISH")

    def detect_three_black_crows(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Three Black Crows pattern.

        3 consecutive long bearish candles, each closing lower than the previous,
        opening within the prior candle body, with relatively short lower shadows.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 3:
                return self._empty_result("Three Black Crows", "BEARISH")

            bars = [clean_df.iloc[-3], clean_df.iloc[-2], clean_df.iloc[-1]]
            opens = [float(b["open"]) for b in bars]
            highs = [float(b["high"]) for b in bars]
            lows = [float(b["low"]) for b in bars]
            closes = [float(b["close"]) for b in bars]

            all_bearish = all(closes[i] < opens[i] for i in range(3))
            descending_closes = (closes[2] < closes[1] < closes[0])
            valid_opens = (
                (opens[1] <= opens[0] and opens[1] >= closes[0])
                and (opens[2] <= opens[1] and opens[2] >= closes[1])
            )
            short_lower_wicks = all(
                (closes[i] - lows[i]) / max(highs[i] - lows[i], 1e-6) <= 0.25 for i in range(3)
            )

            if all_bearish and descending_closes and valid_opens and short_lower_wicks:
                base_conf = 75.0
                if all((closes[i] - lows[i]) / max(highs[i] - lows[i], 1e-6) <= 0.15 for i in range(3)):
                    base_conf += 10.0
                vols = [float(b["volume"]) for b in bars]
                if vols[2] >= vols[1] >= vols[0]:
                    base_conf += 12.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Three Black Crows",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Three Black Crows"],
                    "description": (
                        f"Three Black Crows detected: Three consecutive strong red candles "
                        f"closed at ₹{closes[0]:.2f}, ₹{closes[1]:.2f}, and ₹{closes[2]:.2f} near their lows."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_three_black_crows: {e}")

        return self._empty_result("Three Black Crows", "BEARISH")

    def detect_spinning_top(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Spinning Top pattern.

        Small real body centered between upper and lower shadows of roughly equal length,
        indicating market indecision and equilibrium.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            is_spinning = (
                (body / rng <= 0.30)
                and (upper_shadow > body)
                and (lower_shadow > body)
                and (abs(upper_shadow - lower_shadow) / rng <= 0.30)
            )

            if is_spinning:
                base_conf = 52.0
                if abs(upper_shadow - lower_shadow) / rng <= 0.12:
                    base_conf += 15.0
                if body / rng <= 0.15:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Spinning Top",
                    "direction": "NEUTRAL",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Spinning Top"],
                    "description": (
                        f"Spinning Top detected at ₹{cl:.2f}: Small body ({body/rng*100:.1f}%) centered "
                        f"between upper (₹{h:.2f}) and lower (₹{l:.2f}) shadows, showing market balance."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_spinning_top: {e}")

        return self._empty_result("Spinning Top", "NEUTRAL")

    def detect_marubozu_bullish(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bullish Marubozu pattern.

        Long green candle with virtually no upper or lower shadows (body >= 88% of range),
        showing overwhelming buyer conviction from open to close.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            is_bullish_marubozu = (
                (cl > o)
                and (body / rng >= 0.88)
                and (upper_shadow / rng <= 0.06)
                and (lower_shadow / rng <= 0.06)
            )

            if is_bullish_marubozu:
                base_conf = 65.0
                if body / rng >= 0.95:
                    base_conf += 15.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean() * 1.3:
                    base_conf += 15.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bullish Marubozu",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bullish Marubozu"],
                    "description": (
                        f"Bullish Marubozu detected at ₹{cl:.2f}: Full green body constitutes "
                        f"{body/rng*100:.1f}% of range with no significant wicks, showing pure buying force."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_marubozu_bullish: {e}")

        return self._empty_result("Bullish Marubozu", "BULLISH")

    def detect_marubozu_bearish(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bearish Marubozu pattern.

        Long red candle with virtually no upper or lower shadows (body >= 88% of range),
        showing relentless seller domination from open to close.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            c = clean_df.iloc[-1]
            o, h, l, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
            rng = max(h - l, 1e-6)
            body = abs(cl - o)
            upper_shadow = h - max(o, cl)
            lower_shadow = min(o, cl) - l

            is_bearish_marubozu = (
                (cl < o)
                and (body / rng >= 0.88)
                and (upper_shadow / rng <= 0.06)
                and (lower_shadow / rng <= 0.06)
            )

            if is_bearish_marubozu:
                base_conf = 65.0
                if body / rng >= 0.95:
                    base_conf += 15.0
                if len(clean_df) >= 20 and float(c["volume"]) > clean_df["volume"].iloc[-20:].mean() * 1.3:
                    base_conf += 15.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bearish Marubozu",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bearish Marubozu"],
                    "description": (
                        f"Bearish Marubozu detected at ₹{cl:.2f}: Full red body constitutes "
                        f"{body/rng*100:.1f}% of range with no significant wicks, showing heavy dumping."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_marubozu_bearish: {e}")

        return self._empty_result("Bearish Marubozu", "BEARISH")

    def detect_bullish_harami(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bullish Harami pattern.

        2-bar pattern: Bar 1 is a long red candle; Bar 2 is a small green/neutral candle
        completely contained within the real body of Bar 1.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Bullish Harami", "BULLISH")

            c1, c2 = clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            rng1 = max(h1 - l1, 1e-6)
            body1 = abs(cl1 - o1)
            body2 = abs(cl2 - o2)

            is_bear1 = (cl1 < o1) and (body1 / rng1 >= 0.45)
            contained = (min(o2, cl2) >= cl1) and (max(o2, cl2) <= o1) and (body2 <= 0.60 * body1)

            if is_bear1 and contained:
                base_conf = 55.0
                if body2 <= 0.35 * body1:
                    base_conf += 15.0
                if self._detect_prior_trend(clean_df) == "DOWN":
                    base_conf += 12.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bullish Harami",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bullish Harami"],
                    "description": (
                        f"Bullish Harami detected: Small candle (₹{min(o2,cl2):.2f}-₹{max(o2,cl2):.2f}) "
                        f"contained inside previous large red candle (₹{cl1:.2f}-₹{o1:.2f}), signaling selling stall."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_bullish_harami: {e}")

        return self._empty_result("Bullish Harami", "BULLISH")

    def detect_bearish_harami(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bearish Harami pattern.

        2-bar pattern: Bar 1 is a long green candle; Bar 2 is a small red/neutral candle
        completely contained within the real body of Bar 1.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Bearish Harami", "BEARISH")

            c1, c2 = clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            rng1 = max(h1 - l1, 1e-6)
            body1 = abs(cl1 - o1)
            body2 = abs(cl2 - o2)

            is_bull1 = (cl1 > o1) and (body1 / rng1 >= 0.45)
            contained = (min(o2, cl2) >= o1) and (max(o2, cl2) <= cl1) and (body2 <= 0.60 * body1)

            if is_bull1 and contained:
                base_conf = 55.0
                if body2 <= 0.35 * body1:
                    base_conf += 15.0
                if self._detect_prior_trend(clean_df) == "UP":
                    base_conf += 12.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Bearish Harami",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Bearish Harami"],
                    "description": (
                        f"Bearish Harami detected: Small candle (₹{min(o2,cl2):.2f}-₹{max(o2,cl2):.2f}) "
                        f"contained inside previous large green candle (₹{o1:.2f}-₹{cl1:.2f}), signaling upside exhaustion."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_bearish_harami: {e}")

        return self._empty_result("Bearish Harami", "BEARISH")

    def detect_piercing_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Piercing Pattern.

        2-bar bullish reversal: Bar 1 is long red; Bar 2 opens below Bar 1 low/close
        and surges to close above the 50% midpoint of Bar 1's body, without fully engulfing.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Piercing Pattern", "BULLISH")

            c1, c2 = clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            body1 = abs(cl1 - o1)
            rng1 = max(h1 - l1, 1e-6)
            is_bear1 = (cl1 < o1) and (body1 / rng1 >= 0.45)
            midpoint1 = (o1 + cl1) / 2.0

            is_piercing = (
                is_bear1
                and (cl2 > o2)
                and (o2 < cl1)
                and (cl2 > midpoint1)
                and (cl2 <= o1)
            )

            if is_piercing:
                base_conf = 64.0
                if cl2 >= (cl1 + 0.65 * body1):
                    base_conf += 12.0
                if float(c2["volume"]) > float(c1["volume"]):
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Piercing Pattern",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Piercing Pattern"],
                    "description": (
                        f"Piercing Pattern detected: Green candle opened below prior low at ₹{o2:.2f} "
                        f"and closed at ₹{cl2:.2f}, piercing above midpoint ₹{midpoint1:.2f}."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_piercing_pattern: {e}")

        return self._empty_result("Piercing Pattern", "BULLISH")

    def detect_dark_cloud_cover(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Dark Cloud Cover pattern.

        2-bar bearish reversal: Bar 1 is long green; Bar 2 opens above Bar 1 high/close
        and plunges to close below the 50% midpoint of Bar 1's body, without fully engulfing.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            if len(clean_df) < 2:
                return self._empty_result("Dark Cloud Cover", "BEARISH")

            c1, c2 = clean_df.iloc[-2], clean_df.iloc[-1]
            o1, h1, l1, cl1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"])
            o2, h2, l2, cl2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"])

            body1 = abs(cl1 - o1)
            rng1 = max(h1 - l1, 1e-6)
            is_bull1 = (cl1 > o1) and (body1 / rng1 >= 0.45)
            midpoint1 = (o1 + cl1) / 2.0

            is_dark_cloud = (
                is_bull1
                and (cl2 < o2)
                and (o2 > cl1)
                and (cl2 < midpoint1)
                and (cl2 >= o1)
            )

            if is_dark_cloud:
                base_conf = 60.0
                if cl2 <= (o1 + 0.35 * body1):
                    base_conf += 12.0
                if float(c2["volume"]) > float(c1["volume"]):
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Dark Cloud Cover",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Dark Cloud Cover"],
                    "description": (
                        f"Dark Cloud Cover detected: Red candle opened at ₹{o2:.2f} above prior close "
                        f"and plunged to ₹{cl2:.2f}, penetrating below midpoint ₹{midpoint1:.2f}."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_dark_cloud_cover: {e}")

        return self._empty_result("Dark Cloud Cover", "BEARISH")

    # =========================================================================
    # SECTION 2: CHART PATTERNS (Rolling window 60-120 bars)
    # =========================================================================

    def detect_head_and_shoulders(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Head & Shoulders top pattern.

        Identifies 3 prominent peaks where the middle peak (Head) is distinctly higher
        than the Left and Right Shoulders, and shoulders are within 8% height tolerance.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 30:
                return self._empty_result("Head & Shoulders", "BEARISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            peaks, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(peaks) < 3:
                return self._empty_result("Head & Shoulders", "BEARISH")

            # Check peak triplets (p1, p2, p3) among recent peaks
            recent_peaks = peaks[-6:] if len(peaks) >= 6 else peaks
            for i in range(len(recent_peaks) - 2):
                p1, p2, p3 = int(recent_peaks[i]), int(recent_peaks[i + 1]), int(recent_peaks[i + 2])

                # Time spacing and recency check
                if (p2 - p1 < 5) or (p3 - p2 < 5) or (p3 < n - 25):
                    continue

                h1, h2, h3 = highs[p1], highs[p2], highs[p3]

                # Head must be strictly higher than both shoulders
                if not (h2 >= h1 * 1.015 and h2 >= h3 * 1.015):
                    continue

                # Shoulders must be within 8% of each other
                shoulder_diff = abs(h1 - h3) / max(h1, h3)
                if shoulder_diff > 0.08:
                    continue

                # Locate neckline troughs between shoulders
                t1_range = lows[p1 : p2 + 1]
                t2_range = lows[p2 : p3 + 1]
                if len(t1_range) == 0 or len(t2_range) == 0:
                    continue

                t1_idx = p1 + int(np.argmin(t1_range))
                t2_idx = p2 + int(np.argmin(t2_range))
                v1, v2 = lows[t1_idx], lows[t2_idx]

                neckline_slope = (v2 - v1) / max(t2_idx - t1_idx, 1)
                current_neckline = v1 + neckline_slope * (n - 1 - t1_idx)

                # Current price testing or below neckline
                if closes[-1] <= current_neckline * 1.03:
                    base_conf = 65.0
                    if shoulder_diff <= 0.03:
                        base_conf += 15.0
                    if vols[p2] < vols[p1] and vols[p3] < vols[p2]:  # Volume dry-up on right shoulder
                        base_conf += 10.0
                    if closes[-1] < current_neckline:  # Confirmed neckline breakdown
                        base_conf += 10.0
                    confidence = float(min(100.0, max(0.0, base_conf)))

                    return {
                        "detected": True,
                        "pattern": "Head & Shoulders",
                        "direction": "BEARISH",
                        "confidence": round(confidence, 1),
                        "historical_win_rate": BULKOWSKI_WIN_RATES["Head & Shoulders"],
                        "description": (
                            f"Head & Shoulders top detected: Left Shoulder at ₹{h1:.2f}, Head at ₹{h2:.2f}, "
                            f"Right Shoulder at ₹{h3:.2f}. Neckline support at ₹{current_neckline:.2f}."
                        ),
                    }
        except Exception as e:
            logger.error(f"Error in detect_head_and_shoulders: {e}")

        return self._empty_result("Head & Shoulders", "BEARISH")

    def detect_inverse_head_and_shoulders(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Inverse Head & Shoulders bottom pattern.

        Identifies 3 prominent troughs where the middle trough (Head) is distinctly lower
        than the Left and Right Shoulders, and shoulders are within 8% depth tolerance.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 30:
                return self._empty_result("Inverse Head & Shoulders", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            n = len(w_df)

            peaks, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(troughs) < 3:
                return self._empty_result("Inverse Head & Shoulders", "BULLISH")

            recent_troughs = troughs[-6:] if len(troughs) >= 6 else troughs
            for i in range(len(recent_troughs) - 2):
                t1, t2, t3 = int(recent_troughs[i]), int(recent_troughs[i + 1]), int(recent_troughs[i + 2])

                if (t2 - t1 < 5) or (t3 - t2 < 5) or (t3 < n - 25):
                    continue

                l1, l2, l3 = lows[t1], lows[t2], lows[t3]

                # Head trough must be strictly lower than both shoulders
                if not (l2 <= l1 * 0.985 and l2 <= l3 * 0.985):
                    continue

                shoulder_diff = abs(l1 - l3) / min(l1, l3)
                if shoulder_diff > 0.08:
                    continue

                p1_range = highs[t1 : t2 + 1]
                p2_range = highs[t2 : t3 + 1]
                if len(p1_range) == 0 or len(p2_range) == 0:
                    continue

                p1_idx = t1 + int(np.argmax(p1_range))
                p2_idx = t2 + int(np.argmax(p2_range))
                v1, v2 = highs[p1_idx], highs[p2_idx]

                neckline_slope = (v2 - v1) / max(p2_idx - p1_idx, 1)
                current_neckline = v1 + neckline_slope * (n - 1 - p1_idx)

                if closes[-1] >= current_neckline * 0.97:
                    base_conf = 65.0
                    if shoulder_diff <= 0.03:
                        base_conf += 15.0
                    if closes[-1] > current_neckline:
                        base_conf += 12.0
                    confidence = float(min(100.0, max(0.0, base_conf)))

                    return {
                        "detected": True,
                        "pattern": "Inverse Head & Shoulders",
                        "direction": "BULLISH",
                        "confidence": round(confidence, 1),
                        "historical_win_rate": BULKOWSKI_WIN_RATES["Inverse Head & Shoulders"],
                        "description": (
                            f"Inverse Head & Shoulders bottom detected: Left Shoulder at ₹{l1:.2f}, "
                            f"Head at ₹{l2:.2f}, Right Shoulder at ₹{l3:.2f}. Resistance neckline at ₹{current_neckline:.2f}."
                        ),
                    }
        except Exception as e:
            logger.error(f"Error in detect_inverse_head_and_shoulders: {e}")

        return self._empty_result("Inverse Head & Shoulders", "BULLISH")

    def detect_double_top(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Double Top pattern.

        Identifies two distinct peaks separated by a trough, with peak prices within
        a 3% tolerance of each other.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 25:
                return self._empty_result("Double Top", "BEARISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            peaks, _ = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(peaks) < 2:
                return self._empty_result("Double Top", "BEARISH")

            recent_peaks = peaks[-5:] if len(peaks) >= 5 else peaks
            for i in range(len(recent_peaks) - 1):
                p1, p2 = int(recent_peaks[i]), int(recent_peaks[i + 1])

                # Separation and recency checks
                if (p2 - p1 < 7) or (p2 < n - 25):
                    continue

                h1, h2 = highs[p1], highs[p2]
                diff_pct = abs(h1 - h2) / max(h1, h2)

                # Within 3% tolerance rule
                if diff_pct <= 0.03:
                    # Intermediate valley
                    valley_range = lows[p1 : p2 + 1]
                    if len(valley_range) == 0:
                        continue
                    valley_low = float(np.min(valley_range))

                    # Must have sufficient depth (at least 2%)
                    if (max(h1, h2) - valley_low) / max(h1, h2) < 0.02:
                        continue

                    # Current price is below peak and testing/breaking neckline
                    if closes[-1] <= max(h1, h2):
                        base_conf = 62.0
                        if diff_pct <= 0.01:
                            base_conf += 18.0
                        elif diff_pct <= 0.02:
                            base_conf += 10.0
                        if vols[p2] < vols[p1]:  # Declining volume on second peak
                            base_conf += 10.0
                        if closes[-1] <= valley_low * 1.01:
                            base_conf += 10.0
                        confidence = float(min(100.0, max(0.0, base_conf)))

                        return {
                            "detected": True,
                            "pattern": "Double Top",
                            "direction": "BEARISH",
                            "confidence": round(confidence, 1),
                            "historical_win_rate": BULKOWSKI_WIN_RATES["Double Top"],
                            "description": (
                                f"Double Top detected: Peak 1 at ₹{h1:.2f} and Peak 2 at ₹{h2:.2f} "
                                f"(diff {diff_pct*100:.1f}%), intermediate valley support at ₹{valley_low:.2f}."
                            ),
                        }
        except Exception as e:
            logger.error(f"Error in detect_double_top: {e}")

        return self._empty_result("Double Top", "BEARISH")

    def detect_double_bottom(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Double Bottom pattern.

        Identifies two distinct troughs separated by a peak, with trough prices within
        a 3% tolerance of each other.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 25:
                return self._empty_result("Double Bottom", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            _, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(troughs) < 2:
                return self._empty_result("Double Bottom", "BULLISH")

            recent_troughs = troughs[-5:] if len(troughs) >= 5 else troughs
            for i in range(len(recent_troughs) - 1):
                t1, t2 = int(recent_troughs[i]), int(recent_troughs[i + 1])

                if (t2 - t1 < 7) or (t2 < n - 25):
                    continue

                l1, l2 = lows[t1], lows[t2]
                diff_pct = abs(l1 - l2) / min(l1, l2)

                # Within 3% tolerance rule
                if diff_pct <= 0.03:
                    peak_range = highs[t1 : t2 + 1]
                    if len(peak_range) == 0:
                        continue
                    peak_high = float(np.max(peak_range))

                    if (peak_high - min(l1, l2)) / min(l1, l2) < 0.02:
                        continue

                    if closes[-1] >= min(l1, l2):
                        base_conf = 62.0
                        if diff_pct <= 0.01:
                            base_conf += 18.0
                        elif diff_pct <= 0.02:
                            base_conf += 10.0
                        if vols[t2] < vols[t1]:
                            base_conf += 10.0
                        if closes[-1] >= peak_high * 0.99:
                            base_conf += 10.0
                        confidence = float(min(100.0, max(0.0, base_conf)))

                        return {
                            "detected": True,
                            "pattern": "Double Bottom",
                            "direction": "BULLISH",
                            "confidence": round(confidence, 1),
                            "historical_win_rate": BULKOWSKI_WIN_RATES["Double Bottom"],
                            "description": (
                                f"Double Bottom detected: Trough 1 at ₹{l1:.2f} and Trough 2 at ₹{l2:.2f} "
                                f"(diff {diff_pct*100:.1f}%), resistance neckline at ₹{peak_high:.2f}."
                            ),
                        }
        except Exception as e:
            logger.error(f"Error in detect_double_bottom: {e}")

        return self._empty_result("Double Bottom", "BULLISH")

    def detect_ascending_triangle(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Ascending Triangle pattern.

        Upper horizontal resistance trendline (peaks within 2% range, near-zero slope)
        converging with an ascending support trendline (higher lows).
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 30:
                return self._empty_result("Ascending Triangle", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values

            peaks, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(peaks) < 2 or len(troughs) < 2:
                return self._empty_result("Ascending Triangle", "BULLISH")

            recent_peaks = peaks[-4:]
            recent_troughs = troughs[-4:]

            peak_vals = highs[recent_peaks]
            trough_vals = lows[recent_troughs]

            # Flat horizontal resistance check (<= 2% variance)
            peak_mean = np.mean(peak_vals)
            peak_spread = (np.max(peak_vals) - np.min(peak_vals)) / max(peak_mean, 1e-6)

            slope_p, _, _ = self._fit_trendline(recent_peaks, peak_vals)
            normalized_slope_p = slope_p / max(peak_mean, 1e-6)

            # Ascending support check
            trough_mean = np.mean(trough_vals)
            slope_t, _, r2_t = self._fit_trendline(recent_troughs, trough_vals)
            normalized_slope_t = slope_t / max(trough_mean, 1e-6)

            is_flat_resistance = peak_spread <= 0.02 and abs(normalized_slope_p) < 0.0015
            is_rising_support = normalized_slope_t > 0.0006 and r2_t >= 0.50

            if is_flat_resistance and is_rising_support:
                base_conf = 65.0
                if len(recent_peaks) >= 3:
                    base_conf += 10.0
                if r2_t >= 0.75:
                    base_conf += 12.0
                if closes[-1] >= peak_mean * 0.985:  # Pressing resistance
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Ascending Triangle",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Ascending Triangle"],
                    "description": (
                        f"Ascending Triangle detected: Horizontal resistance at ₹{peak_mean:.2f} "
                        f"paired with rising support (slope +{normalized_slope_t*100:.2f}%/bar), building bullish pressure."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_ascending_triangle: {e}")

        return self._empty_result("Ascending Triangle", "BULLISH")

    def detect_descending_triangle(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Descending Triangle pattern.

        Lower horizontal support trendline (troughs within 2% range, near-zero slope)
        converging with a descending resistance trendline (lower highs).
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 30:
                return self._empty_result("Descending Triangle", "BEARISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values

            peaks, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(peaks) < 2 or len(troughs) < 2:
                return self._empty_result("Descending Triangle", "BEARISH")

            recent_peaks = peaks[-4:]
            recent_troughs = troughs[-4:]

            peak_vals = highs[recent_peaks]
            trough_vals = lows[recent_troughs]

            # Flat horizontal support check
            trough_mean = np.mean(trough_vals)
            trough_spread = (np.max(trough_vals) - np.min(trough_vals)) / max(trough_mean, 1e-6)
            slope_t, _, _ = self._fit_trendline(recent_troughs, trough_vals)
            normalized_slope_t = slope_t / max(trough_mean, 1e-6)

            # Descending resistance check
            peak_mean = np.mean(peak_vals)
            slope_p, _, r2_p = self._fit_trendline(recent_peaks, peak_vals)
            normalized_slope_p = slope_p / max(peak_mean, 1e-6)

            is_flat_support = trough_spread <= 0.02 and abs(normalized_slope_t) < 0.0015
            is_falling_resistance = normalized_slope_p < -0.0006 and r2_p >= 0.50

            if is_flat_support and is_falling_resistance:
                base_conf = 65.0
                if len(recent_troughs) >= 3:
                    base_conf += 10.0
                if r2_p >= 0.75:
                    base_conf += 12.0
                if closes[-1] <= trough_mean * 1.015:
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Descending Triangle",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Descending Triangle"],
                    "description": (
                        f"Descending Triangle detected: Horizontal support at ₹{trough_mean:.2f} "
                        f"pressured by falling resistance (slope {normalized_slope_p*100:.2f}%/bar), signaling breakdown."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_descending_triangle: {e}")

        return self._empty_result("Descending Triangle", "BEARISH")

    def detect_symmetrical_triangle(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Symmetrical Triangle pattern.

        Upper trendline has negative slope (lower highs) and lower trendline has positive
        slope (higher lows), converging towards an apex with contracting volatility.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 30:
                return self._empty_result("Symmetrical Triangle", "NEUTRAL")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values

            peaks, troughs = self._find_peaks_and_troughs(highs, lows, order=4)
            if len(peaks) < 2 or len(troughs) < 2:
                return self._empty_result("Symmetrical Triangle", "NEUTRAL")

            recent_peaks = peaks[-4:]
            recent_troughs = troughs[-4:]

            peak_vals = highs[recent_peaks]
            trough_vals = lows[recent_troughs]

            slope_p, _, r2_p = self._fit_trendline(recent_peaks, peak_vals)
            slope_t, _, r2_t = self._fit_trendline(recent_troughs, trough_vals)

            norm_slope_p = slope_p / max(np.mean(peak_vals), 1e-6)
            norm_slope_t = slope_t / max(np.mean(trough_vals), 1e-6)

            # Symmetrical convergence: upper slopes down, lower slopes up
            is_converging = (
                (norm_slope_p < -0.0006)
                and (norm_slope_t > 0.0006)
                and (r2_p >= 0.45)
                and (r2_t >= 0.45)
            )

            if is_converging:
                prior_trend = self._detect_prior_trend(clean_df, bars_back=15)
                direction = "BULLISH" if prior_trend == "UP" else ("BEARISH" if prior_trend == "DOWN" else "NEUTRAL")

                base_conf = 60.0
                if r2_p >= 0.70 and r2_t >= 0.70:
                    base_conf += 15.0
                if (highs[-1] - lows[-1]) < (highs[0] - lows[0]) * 0.7:  # Volatility squeeze
                    base_conf += 12.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Symmetrical Triangle",
                    "direction": direction,
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Symmetrical Triangle"],
                    "description": (
                        f"Symmetrical Triangle detected: Converging resistance ({norm_slope_p*100:.2f}%/bar) "
                        f"and support (+{norm_slope_t*100:.2f}%/bar), indicating volatility squeeze before breakout."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_symmetrical_triangle: {e}")

        return self._empty_result("Symmetrical Triangle", "NEUTRAL")

    def detect_bull_flag(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Detect Bull Flag pattern.

        Sharp flagpole advance (>= 5% over 5-15 bars) followed by a compact,
        downward-sloping or sideways consolidation channel retracing <= 40% of the pole.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(40, min(80, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 20:
                return self._empty_result("Bull Flag", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            # Flag consolidation is in the last 5 to 15 bars
            flag_len = 8
            pole_end_idx = n - flag_len
            pole_start_idx = max(0, pole_end_idx - 12)

            pole_low = float(np.min(lows[pole_start_idx:pole_end_idx]))
            pole_high = float(np.max(highs[pole_start_idx:pole_end_idx]))
            pole_gain = (pole_high - pole_low) / max(pole_low, 1e-6)

            # Flagpole condition: >= 5% rally
            if pole_gain >= 0.05:
                flag_lows = lows[pole_end_idx:]
                pullback = (pole_high - float(np.min(flag_lows))) / max(pole_high - pole_low, 1e-6)

                # Flag retracement must be shallow (<= 42%)
                if 0.0 <= pullback <= 0.42:
                    slope, _, _ = self._fit_trendline(np.arange(flag_len), closes[pole_end_idx:])
                    norm_slope = slope / max(closes[-1], 1e-6)

                    # Consolidation channel slightly down or flat
                    if -0.006 <= norm_slope <= 0.0015:
                        pole_vol = float(np.mean(vols[pole_start_idx:pole_end_idx]))
                        flag_vol = float(np.mean(vols[pole_end_idx:]))

                        base_conf = 65.0
                        if pullback <= 0.25:
                            base_conf += 15.0
                        if flag_vol < pole_vol * 0.85:  # Volume dry-up in flag
                            base_conf += 12.0
                        confidence = float(min(100.0, max(0.0, base_conf)))

                        return {
                            "detected": True,
                            "pattern": "Bull Flag",
                            "direction": "BULLISH",
                            "confidence": round(confidence, 1),
                            "historical_win_rate": BULKOWSKI_WIN_RATES["Bull Flag"],
                            "description": (
                                f"Bull Flag detected: Flagpole rally of +{pole_gain*100:.1f}% followed by "
                                f"tight consolidation retracing only {pullback*100:.1f}% on lighter volume."
                            ),
                        }
        except Exception as e:
            logger.error(f"Error in detect_bull_flag: {e}")

        return self._empty_result("Bull Flag", "BULLISH")

    def detect_bear_flag(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Detect Bear Flag pattern.

        Sharp flagpole drop (>= 5% over 5-15 bars) followed by a compact,
        upward-sloping or sideways consolidation channel retracing <= 40% of the drop.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(40, min(80, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 20:
                return self._empty_result("Bear Flag", "BEARISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            flag_len = 8
            pole_end_idx = n - flag_len
            pole_start_idx = max(0, pole_end_idx - 12)

            pole_high = float(np.max(highs[pole_start_idx:pole_end_idx]))
            pole_low = float(np.min(lows[pole_start_idx:pole_end_idx]))
            pole_drop = (pole_high - pole_low) / max(pole_high, 1e-6)

            # Flagpole condition: >= 5% drop
            if pole_drop >= 0.05:
                flag_highs = highs[pole_end_idx:]
                bounce = (float(np.max(flag_highs)) - pole_low) / max(pole_high - pole_low, 1e-6)

                # Flag bounce must be weak (<= 42%)
                if 0.0 <= bounce <= 0.42:
                    slope, _, _ = self._fit_trendline(np.arange(flag_len), closes[pole_end_idx:])
                    norm_slope = slope / max(closes[-1], 1e-6)

                    if -0.0015 <= norm_slope <= 0.006:
                        pole_vol = float(np.mean(vols[pole_start_idx:pole_end_idx]))
                        flag_vol = float(np.mean(vols[pole_end_idx:]))

                        base_conf = 65.0
                        if bounce <= 0.25:
                            base_conf += 15.0
                        if flag_vol < pole_vol * 0.85:
                            base_conf += 12.0
                        confidence = float(min(100.0, max(0.0, base_conf)))

                        return {
                            "detected": True,
                            "pattern": "Bear Flag",
                            "direction": "BEARISH",
                            "confidence": round(confidence, 1),
                            "historical_win_rate": BULKOWSKI_WIN_RATES["Bear Flag"],
                            "description": (
                                f"Bear Flag detected: Flagpole drop of -{pole_drop*100:.1f}% followed by "
                                f"weak counter-trend channel bouncing only {bounce*100:.1f}%."
                            ),
                        }
        except Exception as e:
            logger.error(f"Error in detect_bear_flag: {e}")

        return self._empty_result("Bear Flag", "BEARISH")

    def detect_cup_and_handle(
        self,
        df: pd.DataFrame,
        window: int = 100,
    ) -> Dict[str, Any]:
        """
        Detect Cup & Handle pattern.

        U-shaped rounding recovery across 40-100 bars where the left and right rims
        match within 5%, followed by a shallow handle pullback (<= 35% of cup depth).
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(60, min(120, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 45:
                return self._empty_result("Cup & Handle", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            n = len(w_df)

            handle_len = 8
            cup_n = n - handle_len

            # Left rim peak search (bars 0 to 40% of cup)
            left_rim_idx = int(np.argmax(highs[: int(cup_n * 0.40)]))
            left_rim_val = highs[left_rim_idx]

            # Right rim peak search (bars 65% to 100% of cup)
            right_search_start = int(cup_n * 0.65)
            right_rim_rel = int(np.argmax(highs[right_search_start:cup_n]))
            right_rim_idx = right_search_start + right_rim_rel
            right_rim_val = highs[right_rim_idx]

            # Rims within 5% height tolerance
            rim_diff = abs(left_rim_val - right_rim_val) / max(left_rim_val, right_rim_val)
            if rim_diff <= 0.05:
                # Cup bottom between rims
                bottom_rel = int(np.argmin(lows[left_rim_idx:right_rim_idx]))
                bottom_idx = left_rim_idx + bottom_rel
                bottom_val = lows[bottom_idx]
                cup_depth = (left_rim_val - bottom_val) / left_rim_val

                # Cup depth between 6% and 35%
                if 0.06 <= cup_depth <= 0.35:
                    # Roundness check: not an abrupt V-spike (multiple bars in bottom zone)
                    bottom_zone = bottom_val + 0.30 * (left_rim_val - bottom_val)
                    bars_in_bottom = np.sum(lows[left_rim_idx:right_rim_idx] <= bottom_zone)

                    if bars_in_bottom >= 4:
                        # Handle evaluation (last handle_len bars)
                        handle_low = float(np.min(lows[cup_n:]))
                        handle_pullback = (right_rim_val - handle_low) / max(right_rim_val - bottom_val, 1e-6)

                        if handle_pullback <= 0.35 and closes[-1] >= handle_low:
                            base_conf = 70.0
                            if rim_diff <= 0.02:
                                base_conf += 12.0
                            if bars_in_bottom >= 8:  # Very well-rounded bottom
                                base_conf += 10.0
                            confidence = float(min(100.0, max(0.0, base_conf)))

                            return {
                                "detected": True,
                                "pattern": "Cup & Handle",
                                "direction": "BULLISH",
                                "confidence": round(confidence, 1),
                                "historical_win_rate": BULKOWSKI_WIN_RATES["Cup & Handle"],
                                "description": (
                                    f"Cup & Handle detected: U-shaped cup between rims ₹{left_rim_val:.2f} and ₹{right_rim_val:.2f} "
                                    f"(depth {cup_depth*100:.1f}%) with handle consolidation near ₹{closes[-1]:.2f}."
                                ),
                            }
        except Exception as e:
            logger.error(f"Error in detect_cup_and_handle: {e}")

        return self._empty_result("Cup & Handle", "BULLISH")

    def detect_breakout(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Detect Decisive Breakout pattern with volume confirmation.

        Identifies price breaking decisively above multi-bar resistance (or below support)
        with trading volume >= 1.5x of the 20-period moving average.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(30, min(80, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 25:
                return self._empty_result("Breakout", "BULLISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values

            # Resistance & Support established over prior window excluding latest 2 bars
            prior_highs = highs[:-2]
            prior_lows = lows[:-2]
            resistance = float(np.max(prior_highs))
            support = float(np.min(prior_lows))

            vol_sma = float(np.mean(vols[-22:-2]))
            vol_ratio = float(vols[-1] / max(vol_sma, 1.0))

            # Bullish Breakout above resistance with volume
            if closes[-1] >= resistance * 1.003 and vol_ratio >= 1.40:
                base_conf = 65.0
                if vol_ratio >= 2.0:
                    base_conf += 15.0
                if closes[-1] >= (highs[-1] - 0.20 * (highs[-1] - lows[-1])):  # Closed near high
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Breakout",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Breakout"],
                    "description": (
                        f"Bullish Breakout: Price broke above resistance at ₹{resistance:.2f} "
                        f"to ₹{closes[-1]:.2f} confirmed by {vol_ratio:.1f}x average volume."
                    ),
                }

            # Bearish Breakdown below support with volume
            if closes[-1] <= support * 0.997 and vol_ratio >= 1.40:
                base_conf = 65.0
                if vol_ratio >= 2.0:
                    base_conf += 15.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Breakout",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Breakout"],
                    "description": (
                        f"Bearish Breakdown: Price broke below support at ₹{support:.2f} "
                        f"to ₹{closes[-1]:.2f} on heavy volume surge ({vol_ratio:.1f}x average)."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_breakout: {e}")

        return self._empty_result("Breakout", "BULLISH")

    def detect_fakeout(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Detect Fakeout (False Breakout / Trap) pattern.

        Identifies a recent breach of resistance or support that completely failed
        and reversed back into the prior range within 3 bars.
        """
        try:
            clean_df = self._normalize_dataframe(df)
            w = max(30, min(80, window))
            w_df = clean_df.iloc[-min(len(clean_df), w) :].copy().reset_index(drop=True)

            if len(w_df) < 25:
                return self._empty_result("Fakeout", "BEARISH")

            highs = w_df["high"].values
            lows = w_df["low"].values
            closes = w_df["close"].values
            vols = w_df["volume"].values
            n = len(w_df)

            # Key levels established prior to last 4 bars
            prior_highs = highs[:-4]
            prior_lows = lows[:-4]
            resistance = float(np.max(prior_highs))
            support = float(np.min(prior_lows))

            # 1. Bull Fakeout (Bull Trap): Breached above resistance, but collapsed back inside
            had_upside_breach = any(highs[i] >= resistance * 1.002 for i in range(n - 4, n - 1))
            if had_upside_breach and closes[-1] < resistance * 0.998:
                base_conf = 68.0
                if closes[-1] < resistance * 0.992:
                    base_conf += 14.0
                if vols[-1] > np.mean(vols[-20:]):
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Fakeout",
                    "direction": "BEARISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Fakeout"],
                    "description": (
                        f"Bull Fakeout (Bull Trap): Price spiked above resistance at ₹{resistance:.2f} "
                        f"but failed to sustain, collapsing back to ₹{closes[-1]:.2f} within 3 bars."
                    ),
                }

            # 2. Bear Fakeout (Bear Trap): Breached below support, but rebounded back inside
            had_downside_breach = any(lows[i] <= support * 0.998 for i in range(n - 4, n - 1))
            if had_downside_breach and closes[-1] > support * 1.002:
                base_conf = 68.0
                if closes[-1] > support * 1.008:
                    base_conf += 14.0
                if vols[-1] > np.mean(vols[-20:]):
                    base_conf += 10.0
                confidence = float(min(100.0, max(0.0, base_conf)))

                return {
                    "detected": True,
                    "pattern": "Fakeout",
                    "direction": "BULLISH",
                    "confidence": round(confidence, 1),
                    "historical_win_rate": BULKOWSKI_WIN_RATES["Fakeout"],
                    "description": (
                        f"Bear Fakeout (Bear Trap): Price dipped below support at ₹{support:.2f} "
                        f"but bounced back sharply to ₹{closes[-1]:.2f} within 3 bars."
                    ),
                }
        except Exception as e:
            logger.error(f"Error in detect_fakeout: {e}")

        return self._empty_result("Fakeout", "BEARISH")

    # =========================================================================
    # Aggregation & Scanning API
    # =========================================================================

    def scan_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scan and return all detected Candlestick patterns in the dataframe.

        Returns:
            List of detected pattern dictionaries.
        """
        detectors = [
            self.detect_doji,
            self.detect_dragonfly_doji,
            self.detect_gravestone_doji,
            self.detect_hammer,
            self.detect_inverted_hammer,
            self.detect_hanging_man,
            self.detect_shooting_star,
            self.detect_bullish_engulfing,
            self.detect_bearish_engulfing,
            self.detect_morning_star,
            self.detect_evening_star,
            self.detect_three_white_soldiers,
            self.detect_three_black_crows,
            self.detect_spinning_top,
            self.detect_marubozu_bullish,
            self.detect_marubozu_bearish,
            self.detect_bullish_harami,
            self.detect_bearish_harami,
            self.detect_piercing_pattern,
            self.detect_dark_cloud_cover,
        ]

        detected = []
        for detector in detectors:
            try:
                res = detector(df)
                if res.get("detected"):
                    detected.append(res)
            except Exception as e:
                logger.error(f"Error executing {detector.__name__}: {e}")

        return detected

    def scan_chart_patterns(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan and return all detected Chart patterns in the dataframe.

        Args:
            df: Price dataframe.
            window: Rolling window size (60-120 bars, default 100).

        Returns:
            List of detected pattern dictionaries.
        """
        w = window if window is not None else self.default_window
        detectors = [
            lambda d: self.detect_head_and_shoulders(d, window=w),
            lambda d: self.detect_inverse_head_and_shoulders(d, window=w),
            lambda d: self.detect_double_top(d, window=w),
            lambda d: self.detect_double_bottom(d, window=w),
            lambda d: self.detect_ascending_triangle(d, window=w),
            lambda d: self.detect_descending_triangle(d, window=w),
            lambda d: self.detect_symmetrical_triangle(d, window=w),
            lambda d: self.detect_bull_flag(d, window=min(60, w)),
            lambda d: self.detect_bear_flag(d, window=min(60, w)),
            lambda d: self.detect_cup_and_handle(d, window=w),
            lambda d: self.detect_breakout(d, window=min(60, w)),
            lambda d: self.detect_fakeout(d, window=min(60, w)),
        ]

        detected = []
        for detector in detectors:
            try:
                res = detector(df)
                if res.get("detected"):
                    detected.append(res)
            except Exception as e:
                logger.error(f"Error executing chart pattern detector: {e}")

        return detected

    def scan_all_patterns(
        self,
        df: pd.DataFrame,
        window: Optional[int] = None,
        only_detected: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive scan checking ALL Candlestick and Chart patterns.

        Args:
            df: Historical price dataframe with OHLCV data.
            window: Rolling bar window for chart patterns (60-120).
            only_detected: If True, returns only detected patterns; if False, returns all.

        Returns:
            List of pattern result dictionaries.
        """
        cdl_results = self.scan_candlestick_patterns(df)
        chart_results = self.scan_chart_patterns(df, window=window)
        all_detected = cdl_results + chart_results

        if only_detected:
            return all_detected

        # If requested all including non-detected:
        all_pattern_names = list(BULKOWSKI_WIN_RATES.keys())
        detected_names = {p["pattern"] for p in all_detected}
        full_results = list(all_detected)
        for name in all_pattern_names:
            if name not in detected_names:
                full_results.append(self._empty_result(name))

        return full_results

    def get_pattern_summary(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        window: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get an executive pattern analysis summary for UI and API reporting.

        Args:
            df: Price dataframe.
            symbol: Optional stock symbol (e.g. 'RELIANCE.NS').
            window: Rolling bar window for chart pattern analysis.

        Returns:
            Summary dictionary with counts, detected patterns, and SEBI disclaimer.
        """
        detected = self.scan_all_patterns(df, window=window, only_detected=True)
        bullish = [p for p in detected if p.get("direction") == "BULLISH"]
        bearish = [p for p in detected if p.get("direction") == "BEARISH"]
        neutral = [p for p in detected if p.get("direction") == "NEUTRAL"]

        avg_conf = (
            float(np.mean([p["confidence"] for p in detected])) if detected else 0.0
        )

        return {
            "symbol": self.normalize_symbol(symbol) if symbol else None,
            "total_detected": len(detected),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "average_confidence": round(avg_conf, 1),
            "patterns": detected,
            "disclaimer": SEBI_DISCLAIMER,
        }
