"""
Technical Analysis module for Stock AI Analyzer.
Comprehensive indicator computations, multi-timeframe alignment,
relative strength benchmarking against Nifty 50 and sectoral indices,
and weighted composite signal aggregation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pandas_ta as ta

from backend.config import settings
from backend.data.fetcher import StockDataFetcher
from backend.data.nse_symbols import get_sector, get_sector_index

logger = logging.getLogger(__name__)

SEBI_DISCLAIMER: str = getattr(
    settings,
    "disclaimer",
    (
        "⚠️ DISCLAIMER: This is an educational tool for stock market analysis. "
        "It is NOT SEBI-registered investment advice. Past performance does not "
        "guarantee future results. No prediction model is 100% accurate. "
        "Always do your own research and consult a SEBI-registered financial "
        "advisor before making investment decisions. Trade at your own risk."
    ),
)


class TechnicalAnalyzer:
    """
    Production-grade technical analysis engine.
    Calculates trend, momentum, volatility, volume, and relative strength indicators
    using pandas-ta, computes multi-timeframe alignment, and generates weighted
    composite signals with calibrated confidence scores (0-100%).
    """

    def __init__(self):
        self.disclaimer = SEBI_DISCLAIMER

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame column names to Title Case, remove duplicates,
        ensure numeric types, and sort chronologically.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        clean = df.copy()

        # Handle column name standardization
        col_map = {}
        for c in clean.columns:
            c_str = str(c).strip().lower()
            if c_str in ("open", "o"):
                col_map[c] = "Open"
            elif c_str in ("high", "h"):
                col_map[c] = "High"
            elif c_str in ("low", "l"):
                col_map[c] = "Low"
            elif c_str in ("close", "c"):
                col_map[c] = "Close"
            elif c_str in ("adj close", "adj_close", "adjusted_close"):
                col_map[c] = "Adj_Close"
            elif c_str in ("volume", "v", "vol"):
                col_map[c] = "Volume"

        clean = clean.rename(columns=col_map)

        # Standardize any remaining column names
        clean.columns = [str(c).title().replace(" ", "_") for c in clean.columns]

        # Convert OHLCV columns to numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in clean.columns:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean = clean.dropna(subset=["Close"])
        clean = clean[~clean.index.duplicated(keep="last")]
        return clean.sort_index()

    # ==========================================
    # TREND INDICATORS
    # ==========================================

    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute SMA and EMA for 20, 50, 100, 200 periods.
        Evaluates moving average stacking, golden/death cross, and price position.

        Returns:
            Dict containing indicator values, BUY/SELL/NEUTRAL signal, and confidence (0-100).
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 10:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for moving averages",
            }

        close = clean["Close"]
        current_price = float(close.iloc[-1])
        n = len(clean)

        periods = [20, 50, 100, 200]
        smas: Dict[str, Optional[float]] = {}
        emas: Dict[str, Optional[float]] = {}

        for p in periods:
            if n >= p:
                sma_series = ta.sma(close, length=p)
                ema_series = ta.ema(close, length=p)
                smas[f"sma_{p}"] = round(float(sma_series.iloc[-1]), 2) if sma_series is not None and not pd.isna(sma_series.iloc[-1]) else None
                emas[f"ema_{p}"] = round(float(ema_series.iloc[-1]), 2) if ema_series is not None and not pd.isna(ema_series.iloc[-1]) else None
            else:
                smas[f"sma_{p}"] = None
                emas[f"ema_{p}"] = None

        # Cross and stacking evaluation
        ema20 = emas.get("ema_20")
        ema50 = emas.get("ema_50")
        ema100 = emas.get("ema_100")
        ema200 = emas.get("ema_200")

        bullish_points = 0
        bearish_points = 0
        total_evals = 0

        # Price vs EMAs
        for p_name, val in [("EMA 20", ema20), ("EMA 50", ema50), ("EMA 100", ema100), ("EMA 200", ema200)]:
            if val is not None:
                total_evals += 1
                if current_price > val:
                    bullish_points += 1
                else:
                    bearish_points += 1

        # Check alignment / stacking
        alignment = "MIXED"
        if ema20 and ema50 and ema100 and ema200:
            if current_price > ema20 > ema50 > ema100 > ema200:
                alignment = "STRONG_BULLISH_STACK"
                bullish_points += 3
            elif current_price < ema20 < ema50 < ema100 < ema200:
                alignment = "STRONG_BEARISH_STACK"
                bearish_points += 3
            total_evals += 3

        # Golden Cross & Death Cross (EMA 50 vs EMA 200)
        golden_cross = False
        death_cross = False
        if n >= 205 and ema50 and ema200:
            ema50_series = ta.ema(close, length=50)
            ema200_series = ta.ema(close, length=200)
            if ema50_series is not None and ema200_series is not None:
                prev_50 = float(ema50_series.iloc[-2])
                prev_200 = float(ema200_series.iloc[-2])
                if prev_50 <= prev_200 and ema50 > ema200:
                    golden_cross = True
                    bullish_points += 4
                    total_evals += 4
                elif prev_50 >= prev_200 and ema50 < ema200:
                    death_cross = True
                    bearish_points += 4
                    total_evals += 4

        # Signal determination
        if total_evals == 0:
            signal = "NEUTRAL"
            confidence = 50.0
        else:
            bull_ratio = bullish_points / total_evals
            if bull_ratio >= 0.65:
                signal = "BUY"
                confidence = min(round(55.0 + (bull_ratio - 0.65) * 100.0, 1), 95.0)
            elif bull_ratio <= 0.35:
                signal = "SELL"
                bear_ratio = 1.0 - bull_ratio
                confidence = min(round(55.0 + (bear_ratio - 0.65) * 100.0, 1), 95.0)
            else:
                signal = "NEUTRAL"
                confidence = round(50.0 + abs(bull_ratio - 0.5) * 40.0, 1)

        return {
            "value": {
                "current_price": current_price,
                **smas,
                **emas,
            },
            "signal": signal,
            "confidence": confidence,
            "alignment": alignment,
            "crossovers": {
                "golden_cross_50_200": golden_cross,
                "death_cross_50_200": death_cross,
            },
        }

    def calculate_adx(self, df: pd.DataFrame, length: int = 14) -> Dict[str, Any]:
        """
        Calculate Average Directional Index (ADX) along with +DI and -DI.
        Evaluates trend strength and direction.

        Returns:
            Dict containing ADX, +DI, -DI values, BUY/SELL/NEUTRAL signal, and confidence (0-100).
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length * 2):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for ADX calculation",
            }

        try:
            adx_df = ta.adx(high=clean["High"], low=clean["Low"], close=clean["Close"], length=length)
            if adx_df is None or adx_df.empty:
                raise ValueError("ADX calculation returned empty DataFrame")

            # Extract columns dynamically
            adx_col = [c for c in adx_df.columns if c.startswith("ADX")][0]
            dmp_col = [c for c in adx_df.columns if "DMP" in c or "+DI" in c or "DPI" in c][0]
            dmn_col = [c for c in adx_df.columns if "DMN" in c or "-DI" in c or "DMI" in c][0]

            adx_val = float(adx_df[adx_col].iloc[-1])
            dmp_val = float(adx_df[dmp_col].iloc[-1])
            dmn_val = float(adx_df[dmn_col].iloc[-1])

            # Trend strength classification
            if adx_val < 20.0:
                strength = "ABSENT_OR_WEAK"
            elif adx_val < 25.0:
                strength = "EMERGING"
            elif adx_val < 50.0:
                strength = "STRONG"
            elif adx_val < 75.0:
                strength = "VERY_STRONG"
            else:
                strength = "EXTREMELY_STRONG"

            # Signal logic
            if adx_val >= 25.0:
                if dmp_val > dmn_val:
                    signal = "BUY"
                    confidence = min(round(65.0 + (adx_val - 25.0) * 0.7, 1), 95.0)
                else:
                    signal = "SELL"
                    confidence = min(round(65.0 + (adx_val - 25.0) * 0.7, 1), 95.0)
            elif adx_val >= 20.0:
                if dmp_val > dmn_val:
                    signal = "BUY"
                    confidence = 58.0
                else:
                    signal = "SELL"
                    confidence = 58.0
            else:
                signal = "NEUTRAL"
                confidence = 50.0

            return {
                "value": {
                    "adx": round(adx_val, 2),
                    "plus_di": round(dmp_val, 2),
                    "minus_di": round(dmn_val, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "trend_strength": strength,
                "interpretation": f"ADX is {adx_val:.1f} ({strength}) with {'+DI > -DI' if dmp_val > dmn_val else '-DI > +DI'}.",
            }
        except Exception as e:
            logger.error(f"Error in calculate_adx: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_supertrend(
        self, df: pd.DataFrame, length: int = 10, multiplier: float = 3.0
    ) -> Dict[str, Any]:
        """
        Calculate Supertrend indicator using ATR bands.

        Returns:
            Dict containing supertrend price, direction (1 for buy, -1 for sell),
            distance percentage, BUY/SELL/NEUTRAL signal, and confidence (0-100).
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Supertrend",
            }

        try:
            st_df = ta.supertrend(
                high=clean["High"],
                low=clean["Low"],
                close=clean["Close"],
                length=length,
                multiplier=multiplier,
            )

            if st_df is None or st_df.empty:
                raise ValueError("Supertrend calculation returned empty DataFrame")

            st_col = [c for c in st_df.columns if c.startswith("SUPERT_")][0]
            dir_col = [c for c in st_df.columns if c.startswith("SUPERTd_")][0]

            st_val = float(st_df[st_col].iloc[-1])
            direction = int(st_df[dir_col].iloc[-1])  # 1: Uptrend/Buy, -1: Downtrend/Sell
            current_close = float(clean["Close"].iloc[-1])

            distance_pct = abs(current_close - st_val) / st_val * 100.0

            # Check if there was a recent flip within the last 3 bars
            recent_flip = False
            if len(st_df) >= 3:
                prev_dirs = st_df[dir_col].iloc[-3:-1].tolist()
                if any(d != direction for d in prev_dirs):
                    recent_flip = True

            if direction == 1:
                signal = "BUY"
                base_conf = 78.0
            else:
                signal = "SELL"
                base_conf = 78.0

            if recent_flip:
                confidence = 88.0
            else:
                confidence = min(round(base_conf + min(distance_pct, 5.0) * 1.5, 1), 92.0)

            return {
                "value": {
                    "supertrend": round(st_val, 2),
                    "direction": direction,
                    "current_price": round(current_close, 2),
                    "distance_pct": round(distance_pct, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "is_recent_flip": recent_flip,
                "trend": "BULLISH" if direction == 1 else "BEARISH",
            }
        except Exception as e:
            logger.error(f"Error in calculate_supertrend: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_ichimoku(
        self, df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52
    ) -> Dict[str, Any]:
        """
        Calculate Ichimoku Kinko Hyo Cloud components:
        Tenkan-sen (Conversion Line), Kijun-sen (Base Line),
        Senkou Span A (Leading Span A), Senkou Span B (Leading Span B),
        and Chikou Span (Lagging Span).

        Returns:
            Dict containing Ichimoku lines, cloud state, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (senkou + 10):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": f"Insufficient data for Ichimoku (need >= {senkou + 10} rows)",
            }

        try:
            high = clean["High"]
            low = clean["Low"]
            close = clean["Close"]
            current_close = float(close.iloc[-1])

            # Robust standard mathematical formula for Ichimoku
            tenkan_series = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2.0
            kijun_series = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2.0
            senkou_a_series = (tenkan_series + kijun_series) / 2.0
            senkou_b_series = (high.rolling(window=senkou).max() + low.rolling(window=senkou).min()) / 2.0

            # Current values
            tenkan_val = float(tenkan_series.iloc[-1])
            kijun_val = float(kijun_series.iloc[-1])

            # For current price relative to Cloud, we look at Span A and Span B calculated 26 bars ago
            # which defines the current cloud!
            span_a_current = float(senkou_a_series.iloc[-kijun]) if len(clean) >= kijun else float(senkou_a_series.iloc[-1])
            span_b_current = float(senkou_b_series.iloc[-kijun]) if len(clean) >= kijun else float(senkou_b_series.iloc[-1])

            cloud_top = max(span_a_current, span_b_current)
            cloud_bottom = min(span_a_current, span_b_current)

            # Chikou: compare current close to close 26 bars ago
            close_26_ago = float(close.iloc[-kijun]) if len(clean) >= kijun else current_close

            # Position tests
            price_above_cloud = current_close > cloud_top
            price_below_cloud = current_close < cloud_bottom
            tk_bullish_cross = tenkan_val > kijun_val
            chikou_bullish = current_close > close_26_ago

            bullish_count = sum([price_above_cloud, tk_bullish_cross, chikou_bullish])
            bearish_count = sum([price_below_cloud, not tk_bullish_cross, not chikou_bullish])

            if bullish_count == 3:
                signal = "BUY"
                confidence = 88.0
            elif bullish_count == 2 and price_above_cloud:
                signal = "BUY"
                confidence = 72.0
            elif bearish_count == 3:
                signal = "SELL"
                confidence = 88.0
            elif bearish_count == 2 and price_below_cloud:
                signal = "SELL"
                confidence = 72.0
            else:
                signal = "NEUTRAL"
                confidence = 52.0

            position = "ABOVE_CLOUD" if price_above_cloud else ("BELOW_CLOUD" if price_below_cloud else "INSIDE_CLOUD")

            return {
                "value": {
                    "tenkan_sen": round(tenkan_val, 2),
                    "kijun_sen": round(kijun_val, 2),
                    "senkou_span_a": round(span_a_current, 2),
                    "senkou_span_b": round(span_b_current, 2),
                    "cloud_top": round(cloud_top, 2),
                    "cloud_bottom": round(cloud_bottom, 2),
                    "current_price": round(current_close, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "cloud_position": position,
                "tk_cross": "BULLISH" if tk_bullish_cross else "BEARISH",
                "chikou_status": "BULLISH" if chikou_bullish else "BEARISH",
            }
        except Exception as e:
            logger.error(f"Error in calculate_ichimoku: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    # ==========================================
    # MOMENTUM INDICATORS
    # ==========================================

    def calculate_rsi(self, df: pd.DataFrame, length: int = 14) -> Dict[str, Any]:
        """
        Calculate Relative Strength Index (RSI).
        Evaluates overbought/oversold levels and momentum divergence.

        Returns:
            Dict containing RSI value, previous RSI, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for RSI",
            }

        try:
            rsi_series = ta.rsi(clean["Close"], length=length)
            if rsi_series is None or rsi_series.empty:
                raise ValueError("RSI series returned empty")

            current_rsi = float(rsi_series.iloc[-1])
            prev_rsi = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else current_rsi

            # Reversal / extreme detection
            if current_rsi >= 75.0:
                signal = "SELL"
                confidence = min(round(75.0 + (current_rsi - 75.0) * 0.8, 1), 95.0)
                condition = "EXTREMELY_OVERBOUGHT"
            elif current_rsi >= 70.0:
                signal = "SELL"
                confidence = 72.0
                condition = "OVERBOUGHT"
            elif current_rsi <= 25.0:
                signal = "BUY"
                confidence = min(round(75.0 + (25.0 - current_rsi) * 0.8, 1), 95.0)
                condition = "EXTREMELY_OVERSOLD"
            elif current_rsi <= 30.0:
                signal = "BUY"
                confidence = 72.0
                condition = "OVERSOLD"
            elif prev_rsi < 30.0 and current_rsi > 30.0:
                signal = "BUY"
                confidence = 82.0
                condition = "OVERSOLD_REVERSAL_BOUNCE"
            elif prev_rsi > 70.0 and current_rsi < 70.0:
                signal = "SELL"
                confidence = 82.0
                condition = "OVERBOUGHT_REVERSAL_PULLBACK"
            elif 55.0 <= current_rsi < 70.0:
                signal = "BUY"
                confidence = round(60.0 + (current_rsi - 55.0) * 0.8, 1)
                condition = "BULLISH_MOMENTUM"
            elif 30.0 < current_rsi <= 45.0:
                signal = "SELL"
                confidence = round(60.0 + (45.0 - current_rsi) * 0.8, 1)
                condition = "BEARISH_MOMENTUM"
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                condition = "NEUTRAL_RANGE"

            return {
                "value": {
                    "rsi": round(current_rsi, 2),
                    "rsi_prev": round(prev_rsi, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "condition": condition,
            }
        except Exception as e:
            logger.error(f"Error in calculate_rsi: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9
    ) -> Dict[str, Any]:
        """
        Calculate Moving Average Convergence Divergence (MACD), Signal line, and Histogram.
        Detects histogram crossovers and zero-line crossings.

        Returns:
            Dict with MACD line, Signal line, Histogram, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (slow + signal_period):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for MACD",
            }

        try:
            macd_df = ta.macd(clean["Close"], fast=fast, slow=slow, signal=signal_period)
            if macd_df is None or macd_df.empty:
                raise ValueError("MACD calculation returned empty")

            macd_col = [c for c in macd_df.columns if c.startswith("MACD_")][0]
            hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
            sign_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]

            curr_macd = float(macd_df[macd_col].iloc[-1])
            curr_hist = float(macd_df[hist_col].iloc[-1])
            curr_sign = float(macd_df[sign_col].iloc[-1])

            prev_hist = float(macd_df[hist_col].iloc[-2]) if len(macd_df) > 1 else curr_hist

            # Crossover analysis
            bullish_cross = prev_hist <= 0.0 and curr_hist > 0.0
            bearish_cross = prev_hist >= 0.0 and curr_hist < 0.0

            if bullish_cross:
                signal = "BUY"
                confidence = 86.0
                crossover = "FRESH_BULLISH_CROSSOVER"
            elif bearish_cross:
                signal = "SELL"
                confidence = 86.0
                crossover = "FRESH_BEARISH_CROSSOVER"
            elif curr_hist > 0.0:
                signal = "BUY"
                expanding = curr_hist > prev_hist
                confidence = 74.0 if expanding else 64.0
                crossover = "BULLISH_HISTOGRAM_EXPANDING" if expanding else "BULLISH_HISTOGRAM_CONTRACTING"
            elif curr_hist < 0.0:
                signal = "SELL"
                expanding = abs(curr_hist) > abs(prev_hist)
                confidence = 74.0 if expanding else 64.0
                crossover = "BEARISH_HISTOGRAM_EXPANDING" if expanding else "BEARISH_HISTOGRAM_CONTRACTING"
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                crossover = "NONE"

            return {
                "value": {
                    "macd": round(curr_macd, 2),
                    "signal_line": round(curr_sign, 2),
                    "histogram": round(curr_hist, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "crossover": crossover,
            }
        except Exception as e:
            logger.error(f"Error in calculate_macd: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_stochastic(
        self, df: pd.DataFrame, k: int = 14, d: int = 3, smooth_k: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        Identifies overbought/oversold status and line crosses.

        Returns:
            Dict with %K, %D, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (k + d + smooth_k):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Stochastic",
            }

        try:
            stoch_df = ta.stoch(
                high=clean["High"],
                low=clean["Low"],
                close=clean["Close"],
                k=k,
                d=d,
                smooth_k=smooth_k,
            )
            if stoch_df is None or stoch_df.empty:
                raise ValueError("Stochastic calculation returned empty")

            k_col = [c for c in stoch_df.columns if c.startswith("STOCHk_")][0]
            d_col = [c for c in stoch_df.columns if c.startswith("STOCHd_")][0]

            curr_k = float(stoch_df[k_col].iloc[-1])
            curr_d = float(stoch_df[d_col].iloc[-1])
            prev_k = float(stoch_df[k_col].iloc[-2]) if len(stoch_df) > 1 else curr_k
            prev_d = float(stoch_df[d_col].iloc[-2]) if len(stoch_df) > 1 else curr_d

            bullish_cross = prev_k <= prev_d and curr_k > curr_d
            bearish_cross = prev_k >= prev_d and curr_k < curr_d

            if curr_k < 20.0 and bullish_cross:
                signal = "BUY"
                confidence = 88.0
                condition = "OVERSOLD_BULLISH_CROSSOVER"
            elif curr_k > 80.0 and bearish_cross:
                signal = "SELL"
                confidence = 88.0
                condition = "OVERBOUGHT_BEARISH_CROSSOVER"
            elif curr_k < 20.0:
                signal = "BUY"
                confidence = 75.0
                condition = "OVERSOLD"
            elif curr_k > 80.0:
                signal = "SELL"
                confidence = 75.0
                condition = "OVERBOUGHT"
            elif curr_k > curr_d:
                signal = "BUY"
                confidence = 62.0
                condition = "BULLISH_ALIGNMENT"
            elif curr_k < curr_d:
                signal = "SELL"
                confidence = 62.0
                condition = "BEARISH_ALIGNMENT"
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                condition = "NEUTRAL"

            return {
                "value": {
                    "k": round(curr_k, 2),
                    "d": round(curr_d, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "condition": condition,
            }
        except Exception as e:
            logger.error(f"Error in calculate_stochastic: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_cci(self, df: pd.DataFrame, length: int = 20) -> Dict[str, Any]:
        """
        Calculate Commodity Channel Index (CCI).
        Evaluates cyclical momentum and extreme deviations.

        Returns:
            Dict containing CCI value, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for CCI",
            }

        try:
            cci_series = ta.cci(
                high=clean["High"], low=clean["Low"], close=clean["Close"], length=length
            )
            if cci_series is None or cci_series.empty:
                raise ValueError("CCI series returned empty")

            curr_cci = float(cci_series.iloc[-1])
            prev_cci = float(cci_series.iloc[-2]) if len(cci_series) > 1 else curr_cci

            if curr_cci > 200.0:
                signal = "SELL"
                confidence = 76.0
                zone = "EXTREMELY_OVERBOUGHT"
            elif curr_cci > 100.0:
                signal = "BUY"
                confidence = 72.0
                zone = "STRONG_BULLISH_MOMENTUM"
            elif curr_cci < -200.0:
                signal = "BUY"
                confidence = 76.0
                zone = "EXTREMELY_OVERSOLD"
            elif curr_cci < -100.0:
                signal = "SELL"
                confidence = 72.0
                zone = "STRONG_BEARISH_MOMENTUM"
            elif prev_cci < -100.0 and curr_cci >= -100.0:
                signal = "BUY"
                confidence = 80.0
                zone = "BULLISH_RECOVERY_FROM_OVERSOLD"
            elif prev_cci > 100.0 and curr_cci <= 100.0:
                signal = "SELL"
                confidence = 80.0
                zone = "BEARISH_PULLBACK_FROM_OVERBOUGHT"
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                zone = "NORMAL_RANGE"

            return {
                "value": {"cci": round(curr_cci, 2)},
                "signal": signal,
                "confidence": confidence,
                "zone": zone,
            }
        except Exception as e:
            logger.error(f"Error in calculate_cci: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_roc(self, df: pd.DataFrame, length: int = 12) -> Dict[str, Any]:
        """
        Calculate Rate of Change (ROC).
        Measures percentage velocity of price changes over length periods.

        Returns:
            Dict containing ROC value, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for ROC",
            }

        try:
            roc_series = ta.roc(clean["Close"], length=length)
            if roc_series is None or roc_series.empty:
                raise ValueError("ROC calculation returned empty")

            curr_roc = float(roc_series.iloc[-1])

            if curr_roc > 5.0:
                signal = "BUY"
                confidence = min(round(72.0 + (curr_roc - 5.0) * 1.5, 1), 90.0)
            elif curr_roc > 1.0:
                signal = "BUY"
                confidence = 65.0
            elif curr_roc < -5.0:
                signal = "SELL"
                confidence = min(round(72.0 + (abs(curr_roc) - 5.0) * 1.5, 1), 90.0)
            elif curr_roc < -1.0:
                signal = "SELL"
                confidence = 65.0
            else:
                signal = "NEUTRAL"
                confidence = 50.0

            return {
                "value": {"roc": round(curr_roc, 2)},
                "signal": signal,
                "confidence": confidence,
            }
        except Exception as e:
            logger.error(f"Error in calculate_roc: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    # ==========================================
    # VOLATILITY INDICATORS
    # ==========================================

    def calculate_bollinger_bands(
        self, df: pd.DataFrame, length: int = 20, std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands (Upper, Middle SMA 20, Lower), Bandwidth, and %B.
        Detects volatility squeeze and breakout conditions.

        Returns:
            Dict containing band levels, bandwidth, percent_b, squeeze status,
            BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Bollinger Bands",
            }

        try:
            bb_df = ta.bbands(clean["Close"], length=length, std=std)
            if bb_df is None or bb_df.empty:
                raise ValueError("Bollinger Bands calculation returned empty")

            # Extract columns dynamically
            lower_col = [c for c in bb_df.columns if c.startswith("BBL_")][0]
            mid_col = [c for c in bb_df.columns if c.startswith("BBM_")][0]
            upper_col = [c for c in bb_df.columns if c.startswith("BBU_")][0]
            band_col = [c for c in bb_df.columns if c.startswith("BBB_")][0]
            pct_col = [c for c in bb_df.columns if c.startswith("BBP_")][0]

            curr_lower = float(bb_df[lower_col].iloc[-1])
            curr_mid = float(bb_df[mid_col].iloc[-1])
            curr_upper = float(bb_df[upper_col].iloc[-1])
            curr_bandwidth = float(bb_df[band_col].iloc[-1])
            curr_pct_b = float(bb_df[pct_col].iloc[-1])
            current_close = float(clean["Close"].iloc[-1])

            # Squeeze detection: bandwidth near lookback low
            lookback_min = float(bb_df[band_col].tail(60).min())
            is_squeeze = curr_bandwidth <= (lookback_min * 1.15)

            # Signal logic
            if curr_pct_b < 0.0:
                signal = "BUY"
                confidence = 78.0
                position = "BELOW_LOWER_BAND"
            elif curr_pct_b > 1.0:
                signal = "SELL"
                confidence = 72.0
                position = "ABOVE_UPPER_BAND"
            elif 0.5 <= curr_pct_b <= 0.8:
                signal = "BUY"
                confidence = 62.0
                position = "UPPER_HALF"
            elif 0.2 <= curr_pct_b < 0.5:
                signal = "SELL"
                confidence = 62.0
                position = "LOWER_HALF"
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                position = "MID_BAND"

            return {
                "value": {
                    "upper": round(curr_upper, 2),
                    "middle": round(curr_mid, 2),
                    "lower": round(curr_lower, 2),
                    "bandwidth": round(curr_bandwidth, 2),
                    "percent_b": round(curr_pct_b, 3),
                    "current_price": round(current_close, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "is_squeeze": is_squeeze,
                "position": position,
            }
        except Exception as e:
            logger.error(f"Error in calculate_bollinger_bands: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_atr(self, df: pd.DataFrame, length: int = 14) -> Dict[str, Any]:
        """
        Calculate Average True Range (ATR) and normalized ATR percentage.
        Provides volatility regime classification and suggested stop-loss buffer.

        Returns:
            Dict with ATR value, ATR %, volatility regime, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (length + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for ATR",
            }

        try:
            atr_series = ta.atr(
                high=clean["High"], low=clean["Low"], close=clean["Close"], length=length
            )
            if atr_series is None or atr_series.empty:
                raise ValueError("ATR series returned empty")

            curr_atr = float(atr_series.iloc[-1])
            curr_close = float(clean["Close"].iloc[-1])
            atr_pct = (curr_atr / curr_close) * 100.0

            if atr_pct < 1.5:
                regime = "LOW"
            elif atr_pct <= 3.0:
                regime = "NORMAL"
            else:
                regime = "HIGH"

            return {
                "value": {
                    "atr": round(curr_atr, 2),
                    "atr_pct": round(atr_pct, 2),
                    "suggested_sl_distance": round(curr_atr * 1.5, 2),
                    "suggested_target_distance": round(curr_atr * 3.0, 2),
                },
                "signal": "NEUTRAL",
                "confidence": 65.0,
                "volatility_regime": regime,
            }
        except Exception as e:
            logger.error(f"Error in calculate_atr: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_bollinger_bandwidth(
        self, df: pd.DataFrame, length: int = 20, std: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detailed Bollinger Bandwidth and %B indicator analysis.
        """
        bb = self.calculate_bollinger_bands(df, length=length, std=std)
        if bb.get("error"):
            return bb

        vals = bb["value"]
        bandwidth = vals["bandwidth"]
        percent_b = vals["percent_b"]
        squeeze = bb["is_squeeze"]

        return {
            "value": {
                "bandwidth": bandwidth,
                "percent_b": percent_b,
                "squeeze": squeeze,
            },
            "signal": bb["signal"],
            "confidence": bb["confidence"],
            "squeeze_state": "SQUEEZE_ACTIVE" if squeeze else "NORMAL_EXPANSION",
        }

    # ==========================================
    # VOLUME INDICATORS
    # ==========================================

    def calculate_volume_spike(
        self, df: pd.DataFrame, lookback: int = 20, threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect whether recent volume represents a spike (> 2x of 20-day average).
        Distinguishes institutional accumulation vs institutional distribution.

        Returns:
            Dict containing volume ratio, spike flag, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (lookback + 2):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Volume Spike",
            }

        try:
            vols = clean["Volume"]
            curr_vol = float(vols.iloc[-1])
            avg_vol = float(vols.iloc[-lookback - 1 : -1].mean())

            if avg_vol <= 0:
                ratio = 1.0
            else:
                ratio = curr_vol / avg_vol

            curr_close = float(clean["Close"].iloc[-1])
            curr_open = float(clean["Open"].iloc[-1])
            is_green = curr_close >= curr_open

            is_spike = ratio >= threshold

            if is_spike:
                if is_green:
                    signal = "BUY"
                    confidence = min(round(76.0 + (ratio - threshold) * 8.0, 1), 94.0)
                    interpretation = f"Institutional Accumulation: Volume is {ratio:.1f}x 20-day average on positive close."
                else:
                    signal = "SELL"
                    confidence = min(round(76.0 + (ratio - threshold) * 8.0, 1), 94.0)
                    interpretation = f"Institutional Distribution: Volume is {ratio:.1f}x 20-day average on negative close."
            elif ratio >= 1.3:
                if is_green:
                    signal = "BUY"
                    confidence = 66.0
                    interpretation = f"Above-average buying volume ({ratio:.1f}x)."
                else:
                    signal = "SELL"
                    confidence = 66.0
                    interpretation = f"Above-average selling volume ({ratio:.1f}x)."
            else:
                signal = "NEUTRAL"
                confidence = 50.0
                interpretation = f"Normal volume ({ratio:.1f}x 20-day average)."

            return {
                "value": {
                    "current_volume": int(curr_vol),
                    "avg_volume_20": int(avg_vol),
                    "volume_ratio": round(ratio, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "is_spike": is_spike,
                "interpretation": interpretation,
            }
        except Exception as e:
            logger.error(f"Error in calculate_volume_spike: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate On-Balance Volume (OBV) and evaluate its trend against a 20-period EMA.

        Returns:
            Dict with OBV, OBV EMA, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 25:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for OBV",
            }

        try:
            obv_series = ta.obv(clean["Close"], clean["Volume"])
            if obv_series is None or obv_series.empty:
                raise ValueError("OBV series returned empty")

            obv_ema = ta.ema(obv_series, length=20)
            curr_obv = float(obv_series.iloc[-1])
            curr_ema = float(obv_ema.iloc[-1]) if obv_ema is not None else curr_obv
            prev_obv = float(obv_series.iloc[-5]) if len(obv_series) >= 5 else curr_obv

            slope_positive = curr_obv > prev_obv
            above_ema = curr_obv > curr_ema

            if above_ema and slope_positive:
                signal = "BUY"
                confidence = 76.0
                trend = "STRONG_ACCUMULATION"
            elif not above_ema and not slope_positive:
                signal = "SELL"
                confidence = 76.0
                trend = "STRONG_DISTRIBUTION"
            elif above_ema:
                signal = "BUY"
                confidence = 62.0
                trend = "MODERATE_ACCUMULATION"
            else:
                signal = "SELL"
                confidence = 62.0
                trend = "MODERATE_DISTRIBUTION"

            return {
                "value": {
                    "obv": float(curr_obv),
                    "obv_ema_20": float(curr_ema),
                },
                "signal": signal,
                "confidence": confidence,
                "trend": trend,
            }
        except Exception as e:
            logger.error(f"Error in calculate_obv: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Volume Weighted Average Price (VWAP).
        Uses pandas_ta or fallback robust cumulative VWAP.

        Returns:
            Dict containing VWAP value, current price, distance percentage,
            BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 5:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for VWAP",
            }

        try:
            high = clean["High"]
            low = clean["Low"]
            close = clean["Close"]
            volume = clean["Volume"]
            curr_close = float(close.iloc[-1])

            # Robust standard VWAP calculation: (TP * V) / V
            tp = (high + low + close) / 3.0
            cum_vol = volume.cumsum()
            cum_tp_vol = (tp * volume).cumsum()

            vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
            curr_vwap = float(vwap_series.iloc[-1])

            if pd.isna(curr_vwap) or curr_vwap <= 0:
                curr_vwap = curr_close

            dist_pct = ((curr_close - curr_vwap) / curr_vwap) * 100.0

            if dist_pct > 0.5:
                signal = "BUY"
                confidence = min(round(65.0 + min(dist_pct, 5.0) * 3.0, 1), 88.0)
            elif dist_pct < -0.5:
                signal = "SELL"
                confidence = min(round(65.0 + min(abs(dist_pct), 5.0) * 3.0, 1), 88.0)
            else:
                signal = "NEUTRAL"
                confidence = 52.0

            return {
                "value": {
                    "vwap": round(curr_vwap, 2),
                    "current_price": round(curr_close, 2),
                    "distance_pct": round(dist_pct, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "interpretation": f"Price is {abs(dist_pct):.2f}% {'above' if dist_pct >= 0 else 'below'} VWAP.",
            }
        except Exception as e:
            logger.error(f"Error in calculate_vwap: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_volume_profile(
        self, df: pd.DataFrame, bins: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate Price-Volume Distribution (Volume Profile).
        Identifies Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).

        Returns:
            Dict containing POC, VAH, VAL, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Volume Profile",
            }

        try:
            # Use last 60 candles or all available
            sample = clean.tail(min(len(clean), 60))
            p_min = float(sample["Low"].min())
            p_max = float(sample["High"].max())

            if p_max <= p_min:
                return {
                    "value": {},
                    "signal": "NEUTRAL",
                    "confidence": 0.0,
                    "error": "Invalid price range for volume profile",
                }

            bin_edges = np.linspace(p_min, p_max, bins + 1)
            bin_vols = np.zeros(bins)

            for _, row in sample.iterrows():
                vol = float(row["Volume"])
                low_p = float(row["Low"])
                high_p = float(row["High"])

                # Distribute volume evenly across intersecting bins
                idxs = np.where((bin_edges[1:] >= low_p) & (bin_edges[:-1] <= high_p))[0]
                if len(idxs) > 0:
                    allocated_vol = vol / len(idxs)
                    for i in idxs:
                        bin_vols[i] += allocated_vol

            # Point of Control (POC)
            poc_idx = int(np.argmax(bin_vols))
            poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

            # Value Area: 70% of total volume around POC
            total_vol = float(np.sum(bin_vols))
            target_vol = total_vol * 0.70

            cum_vol = bin_vols[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while cum_vol < target_vol and (low_idx > 0 or high_idx < bins - 1):
                vol_below = bin_vols[low_idx - 1] if low_idx > 0 else 0
                vol_above = bin_vols[high_idx + 1] if high_idx < bins - 1 else 0

                if vol_above >= vol_below and high_idx < bins - 1:
                    high_idx += 1
                    cum_vol += vol_above
                elif low_idx > 0:
                    low_idx -= 1
                    cum_vol += vol_below
                else:
                    break

            vah_price = float(bin_edges[high_idx + 1])
            val_price = float(bin_edges[low_idx])
            curr_close = float(clean["Close"].iloc[-1])

            # Location interpretation
            if curr_close > vah_price:
                signal = "BUY"
                confidence = 76.0
                location = "ABOVE_VALUE_AREA_BREAKOUT"
            elif curr_close < val_price:
                signal = "SELL"
                confidence = 76.0
                location = "BELOW_VALUE_AREA_BREAKDOWN"
            else:
                # Inside value area
                dist_poc_pct = abs(curr_close - poc_price) / poc_price * 100.0
                if dist_poc_pct <= 0.6:
                    signal = "NEUTRAL"
                    confidence = 55.0
                    location = "AT_POINT_OF_CONTROL"
                elif curr_close > poc_price:
                    signal = "BUY"
                    confidence = 62.0
                    location = "INSIDE_VA_ABOVE_POC"
                else:
                    signal = "SELL"
                    confidence = 62.0
                    location = "INSIDE_VA_BELOW_POC"

            return {
                "value": {
                    "poc": round(poc_price, 2),
                    "vah": round(vah_price, 2),
                    "val": round(val_price, 2),
                    "current_price": round(curr_close, 2),
                    "total_profile_volume": int(total_vol),
                },
                "signal": signal,
                "confidence": confidence,
                "location": location,
            }
        except Exception as e:
            logger.error(f"Error in calculate_volume_profile: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    # ==========================================
    # RELATIVE STRENGTH (RS)
    # ==========================================

    def calculate_relative_strength_nifty(
        self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate Relative Strength of stock against Nifty 50 (^NSEI).
        Computes return ratio and Mansfield Relative Strength (MRS).

        Returns:
            Dict with RS ratio, Mansfield RS, 3-month outperformance, BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 30:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient stock data for Relative Strength",
            }

        try:
            if benchmark_df is None or benchmark_df.empty:
                benchmark_df = StockDataFetcher.get_nifty50_data(period="1y")

            bench_clean = self._clean_df(benchmark_df)
            if bench_clean.empty or len(bench_clean) < 30:
                raise ValueError("Nifty 50 benchmark data unavailable")

            # Align on matching dates
            combined = pd.DataFrame(
                {"stock": clean["Close"], "bench": bench_clean["Close"]}
            ).dropna()

            if len(combined) < 20:
                raise ValueError("Insufficient overlapping dates with benchmark")

            # RS Ratio: Stock / Benchmark
            rs_ratio = combined["stock"] / combined["bench"]
            rs_sma_50 = rs_ratio.rolling(min(50, len(rs_ratio))).mean()

            # Mansfield RS: ((RS / SMA(RS)) - 1) * 100
            mrs = ((rs_ratio / rs_sma_50) - 1.0) * 100.0

            curr_mrs = float(mrs.iloc[-1])
            curr_rs = float(rs_ratio.iloc[-1])
            prev_mrs = float(mrs.iloc[-5]) if len(mrs) >= 5 else curr_mrs

            # Calculate 3-month (60-day) percentage return comparison
            lookback_days = min(60, len(combined) - 1)
            stock_ret = float(
                (combined["stock"].iloc[-1] - combined["stock"].iloc[-lookback_days])
                / combined["stock"].iloc[-lookback_days]
                * 100.0
            )
            bench_ret = float(
                (combined["bench"].iloc[-1] - combined["bench"].iloc[-lookback_days])
                / combined["bench"].iloc[-lookback_days]
                * 100.0
            )
            outperformance = stock_ret - bench_ret

            # Signal logic
            if curr_mrs > 0.0 and curr_mrs > prev_mrs:
                signal = "BUY"
                confidence = min(round(72.0 + min(curr_mrs, 10.0) * 1.8, 1), 92.0)
                status = "OUTPERFORMING_EXPANDING"
            elif curr_mrs > 0.0:
                signal = "BUY"
                confidence = 66.0
                status = "OUTPERFORMING_STABLE"
            elif curr_mrs < 0.0 and curr_mrs < prev_mrs:
                signal = "SELL"
                confidence = min(round(72.0 + min(abs(curr_mrs), 10.0) * 1.8, 1), 92.0)
                status = "UNDERPERFORMING_WEAKENING"
            else:
                signal = "SELL"
                confidence = 62.0
                status = "UNDERPERFORMING"

            return {
                "value": {
                    "benchmark": "NIFTY 50",
                    "mansfield_rs": round(curr_mrs, 2),
                    "rs_ratio": round(curr_rs, 4),
                    "stock_return_60d_pct": round(stock_ret, 2),
                    "nifty_return_60d_pct": round(bench_ret, 2),
                    "outperformance_pct": round(outperformance, 2),
                },
                "signal": signal,
                "confidence": confidence,
                "status": status,
            }
        except Exception as e:
            logger.error(f"Error in calculate_relative_strength_nifty: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    def calculate_sector_relative_strength(
        self, df: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """
        Calculate Relative Strength of stock against its specific sectoral index.

        Returns:
            Dict containing sectoral benchmark, RS metrics, BUY/SELL/NEUTRAL signal, and confidence.
        """
        try:
            sector = get_sector(symbol)
            sector_index = get_sector_index(sector)

            sector_df = StockDataFetcher.get_ohlcv(sector_index, period="1y")
            if sector_df.empty:
                logger.info(f"Sector index data for {sector_index} unavailable, falling back to Nifty 50")
                return self.calculate_relative_strength_nifty(df)

            result = self.calculate_relative_strength_nifty(df, benchmark_df=sector_df)
            if "value" in result and result["value"]:
                result["value"]["benchmark"] = f"Sector: {sector} ({sector_index})"
                result["value"]["sector_name"] = sector

            return result
        except Exception as e:
            logger.error(f"Error in calculate_sector_relative_strength: {e}")
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": str(e),
            }

    # ==========================================
    # MULTI-TIMEFRAME ANALYSIS
    # ==========================================

    def analyze_multi_timeframe(
        self,
        symbol: Optional[str] = None,
        timeframes: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple timeframes ('weekly', 'daily', '15min').
        Computes trend direction for each timeframe and calculates alignment score.

        Returns:
            Dict containing timeframe breakdown, alignment score (0-100%),
            composite multi-timeframe direction, and confidence.
        """
        if timeframes is None:
            if not symbol:
                return {
                    "alignment_score": 0.0,
                    "composite_direction": "NEUTRAL",
                    "composite_confidence": 0.0,
                    "error": "Symbol or timeframes dict required",
                }
            timeframes = StockDataFetcher.get_multiple_timeframes(symbol)

        tf_results: Dict[str, Dict[str, Any]] = {}
        tf_weights = {"weekly": 0.40, "daily": 0.40, "15min": 0.20}

        weighted_direction_score = 0.0
        active_weights = 0.0

        for tf_name in ["weekly", "daily", "15min"]:
            raw_tf_df = timeframes.get(tf_name)
            if raw_tf_df is None or raw_tf_df.empty or len(raw_tf_df) < 15:
                tf_results[tf_name] = {
                    "signal": "NEUTRAL",
                    "trend": "UNKNOWN",
                    "confidence": 50.0,
                    "status": "INSUFFICIENT_DATA",
                }
                continue

            tf_df = self._clean_df(raw_tf_df)

            # Evaluate key trend indicators on this timeframe
            ma_res = self.calculate_moving_averages(tf_df)
            st_res = self.calculate_supertrend(tf_df)
            rsi_res = self.calculate_rsi(tf_df)

            signals = [ma_res["signal"], st_res["signal"], rsi_res["signal"]]
            buy_votes = signals.count("BUY")
            sell_votes = signals.count("SELL")

            if buy_votes >= 2:
                tf_dir = "BUY"
                tf_conf = round(np.mean([ma_res["confidence"], st_res["confidence"], rsi_res["confidence"]]), 1)
            elif sell_votes >= 2:
                tf_dir = "SELL"
                tf_conf = round(np.mean([ma_res["confidence"], st_res["confidence"], rsi_res["confidence"]]), 1)
            else:
                tf_dir = "NEUTRAL"
                tf_conf = 50.0

            tf_results[tf_name] = {
                "signal": tf_dir,
                "trend": "UPTREND" if tf_dir == "BUY" else ("DOWNTREND" if tf_dir == "SELL" else "SIDEWAYS"),
                "confidence": tf_conf,
                "close": round(float(tf_df["Close"].iloc[-1]), 2),
            }

            weight = tf_weights.get(tf_name, 0.33)
            active_weights += weight
            dir_val = 1.0 if tf_dir == "BUY" else (-1.0 if tf_dir == "SELL" else 0.0)
            weighted_direction_score += weight * dir_val * (tf_conf / 100.0)

        if active_weights > 0:
            norm_score = (weighted_direction_score / active_weights) * 100.0
        else:
            norm_score = 0.0

        # Alignment setup identification
        w_sig = tf_results.get("weekly", {}).get("signal")
        d_sig = tf_results.get("daily", {}).get("signal")
        m_sig = tf_results.get("15min", {}).get("signal")

        if w_sig == "BUY" and d_sig == "BUY" and m_sig == "BUY":
            setup = "PERFECT_BULLISH_ALIGNMENT"
            alignment_score = 100.0
            direction = "BUY"
            confidence = 92.0
        elif w_sig == "SELL" and d_sig == "SELL" and m_sig == "SELL":
            setup = "PERFECT_BEARISH_ALIGNMENT"
            alignment_score = 100.0
            direction = "SELL"
            confidence = 92.0
        elif w_sig == "BUY" and d_sig == "BUY" and m_sig != "BUY":
            setup = "PULLBACK_IN_MAJOR_UPTREND"
            alignment_score = 80.0
            direction = "BUY"
            confidence = 82.0
        elif w_sig == "SELL" and d_sig == "SELL" and m_sig != "SELL":
            setup = "BOUNCE_IN_MAJOR_DOWNTREND"
            alignment_score = 80.0
            direction = "SELL"
            confidence = 82.0
        elif norm_score >= 25.0:
            setup = "MODERATELY_BULLISH"
            alignment_score = round(abs(norm_score), 1)
            direction = "BUY"
            confidence = min(round(65.0 + norm_score * 0.25, 1), 88.0)
        elif norm_score <= -25.0:
            setup = "MODERATELY_BEARISH"
            alignment_score = round(abs(norm_score), 1)
            direction = "SELL"
            confidence = min(round(65.0 + abs(norm_score) * 0.25, 1), 88.0)
        else:
            setup = "MIXED_OR_CONSOLIDATING"
            alignment_score = round(abs(norm_score), 1)
            direction = "NEUTRAL"
            confidence = 50.0

        return {
            "alignment_score": alignment_score,
            "composite_direction": direction,
            "composite_confidence": confidence,
            "setup": setup,
            "timeframes": tf_results,
        }

    # ==========================================
    # FULL ANALYSIS & COMPOSITE SIGNAL
    # ==========================================

    def run_full_analysis(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute all technical indicators on DataFrame and compute weighted composite signal.

        Args:
            df: OHLCV DataFrame
            symbol: Optional stock symbol for relative strength benchmarking

        Returns:
            Structured dictionary with all indicators, composite direction,
            confidence score (0-100), and SEBI disclaimer.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "composite": {"direction": "NEUTRAL", "confidence": 0.0, "score": 0.0},
                "error": "Insufficient data for full technical analysis",
                "sebi_disclaimer": self.disclaimer,
            }

        # 1. Trend
        ma_res = self.calculate_moving_averages(clean)
        adx_res = self.calculate_adx(clean)
        st_res = self.calculate_supertrend(clean)
        ichi_res = self.calculate_ichimoku(clean)

        # 2. Momentum
        rsi_res = self.calculate_rsi(clean)
        macd_res = self.calculate_macd(clean)
        stoch_res = self.calculate_stochastic(clean)
        cci_res = self.calculate_cci(clean)
        roc_res = self.calculate_roc(clean)

        # 3. Volatility
        bb_res = self.calculate_bollinger_bands(clean)
        atr_res = self.calculate_atr(clean)

        # 4. Volume
        spike_res = self.calculate_volume_spike(clean)
        obv_res = self.calculate_obv(clean)
        vwap_res = self.calculate_vwap(clean)
        profile_res = self.calculate_volume_profile(clean)

        # 5. Relative Strength
        rs_nifty = self.calculate_relative_strength_nifty(clean)
        rs_sector = (
            self.calculate_sector_relative_strength(clean, symbol)
            if symbol
            else {"signal": "NEUTRAL", "confidence": 50.0, "value": {}}
        )

        indicators_map = {
            "moving_averages": (ma_res, 0.12),
            "supertrend": (st_res, 0.12),
            "adx": (adx_res, 0.07),
            "ichimoku": (ichi_res, 0.07),
            "rsi": (rsi_res, 0.10),
            "macd": (macd_res, 0.11),
            "stochastic": (stoch_res, 0.06),
            "cci": (cci_res, 0.04),
            "roc": (roc_res, 0.04),
            "bollinger_bands": (bb_res, 0.07),
            "volume_spike": (spike_res, 0.07),
            "obv": (obv_res, 0.05),
            "vwap": (vwap_res, 0.04),
            "relative_strength_nifty": (rs_nifty, 0.04),
        }

        weighted_score = 0.0
        total_weight = 0.0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for name, (res, weight) in indicators_map.items():
            if not res or res.get("error"):
                continue

            sig = res.get("signal", "NEUTRAL")
            conf = float(res.get("confidence", 50.0))

            if sig == "BUY":
                val = 1.0
                bullish_count += 1
            elif sig == "SELL":
                val = -1.0
                bearish_count += 1
            else:
                val = 0.0
                neutral_count += 1

            weighted_score += weight * val * conf
            total_weight += weight

        if total_weight > 0:
            composite_score = weighted_score / total_weight
        else:
            composite_score = 0.0

        # Classify composite direction and calibrated confidence
        if composite_score >= 20.0:
            direction = "BUY"
            confidence = min(round(55.0 + (composite_score - 20.0) * 0.5 + (bullish_count * 2.0), 1), 95.0)
        elif composite_score <= -20.0:
            direction = "SELL"
            confidence = min(round(55.0 + (abs(composite_score) - 20.0) * 0.5 + (bearish_count * 2.0), 1), 95.0)
        else:
            direction = "NEUTRAL"
            confidence = round(50.0 + abs(composite_score) * 0.4, 1)

        summary_text = (
            f"Technical outlook is {direction} with {confidence:.1f}% confidence "
            f"({bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral indicators)."
        )

        composite_block = {
            "direction": direction,
            "confidence": confidence,
            "score": round(composite_score, 2),
            "summary": summary_text,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
        }

        return {
            "symbol": symbol.upper() if symbol else None,
            "direction": direction,
            "confidence": confidence,
            "composite": composite_block,
            "trend": {
                "moving_averages": ma_res,
                "adx": adx_res,
                "supertrend": st_res,
                "ichimoku": ichi_res,
            },
            "momentum": {
                "rsi": rsi_res,
                "macd": macd_res,
                "stochastic": stoch_res,
                "cci": cci_res,
                "roc": roc_res,
            },
            "volatility": {
                "bollinger_bands": bb_res,
                "atr": atr_res,
            },
            "volume": {
                "volume_spike": spike_res,
                "obv": obv_res,
                "vwap": vwap_res,
                "volume_profile": profile_res,
            },
            "relative_strength": {
                "nifty50": rs_nifty,
                "sector": rs_sector,
            },
            "sebi_disclaimer": self.disclaimer,
        }

    # ==========================================
    # MAIN ENTRY POINT
    # ==========================================

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Main entry point for analyzing a stock by symbol.
        Fetches daily and multi-timeframe data, runs all technical analysis,
        and returns a complete analysis report.

        Args:
            symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS', 'INFY')

        Returns:
            Comprehensive technical analysis dictionary.
        """
        norm_sym = StockDataFetcher.normalize_symbol(symbol)
        clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "")

        logger.info(f"Running full technical analysis for {clean_sym} ({norm_sym})")

        # Fetch daily OHLCV
        df = StockDataFetcher.get_ohlcv(clean_sym, period="2y", interval="1d")
        if df.empty:
            logger.warning(f"No OHLCV data found for {clean_sym}")
            return {
                "symbol": clean_sym,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "composite": {"direction": "NEUTRAL", "confidence": 0.0},
                "error": f"No data returned for symbol {clean_sym}",
                "sebi_disclaimer": self.disclaimer,
            }

        # Run indicator pipeline
        result = self.run_full_analysis(df, symbol=clean_sym)

        # Multi-timeframe analysis
        mtf_data = StockDataFetcher.get_multiple_timeframes(clean_sym)
        mtf_result = self.analyze_multi_timeframe(symbol=clean_sym, timeframes=mtf_data)
        result["multi_timeframe"] = mtf_result

        # Current price summary
        current_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_close
        pct_change = ((current_close - prev_close) / prev_close) * 100.0

        result["current_price"] = round(current_close, 2)
        result["prev_close"] = round(prev_close, 2)
        result["change_pct"] = round(pct_change, 2)

        return result
