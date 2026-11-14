"""
Support and Resistance Analysis module for Stock AI Analyzer.
Implements:
1. Multi-system Pivot Points (Standard, Fibonacci, Camarilla, Woodie)
2. Fibonacci Retracement (23.6%, 38.2%, 50.0%, 61.8%, 78.6%)
3. Fibonacci Extension (100.0%, 127.2%, 161.8%, 200.0%, 261.8%)
4. Dynamic Price Action S/R (Fractal-based swing clustering)
5. Confluence Zone Clustering (<= 1% tolerance across all indicators)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pandas_ta as ta

from backend.config import settings
from backend.data.fetcher import StockDataFetcher

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


class SupportResistanceAnalyzer:
    """
    Advanced Support and Resistance analyzer.
    Detects key structural price levels using classical mathematical pivots,
    Fibonacci ratios, swing fractals, and multi-indicator confluence clustering.
    """

    def __init__(self):
        self.disclaimer = SEBI_DISCLAIMER

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and sanitize data."""
        if df is None or df.empty:
            return pd.DataFrame()

        clean = df.copy()
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
            elif c_str in ("volume", "v", "vol"):
                col_map[c] = "Volume"

        clean = clean.rename(columns=col_map)
        clean.columns = [str(c).title().replace(" ", "_") for c in clean.columns]

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in clean.columns:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean = clean.dropna(subset=["Close", "High", "Low"])
        clean = clean[~clean.index.duplicated(keep="last")]
        return clean.sort_index()

    # ==========================================
    # 1. PIVOT POINTS
    # ==========================================

    @staticmethod
    def calculate_standard_pivots(
        high: float, low: float, close: float
    ) -> Dict[str, float]:
        """
        Classic / Standard Pivot Points.
        P = (H + L + C) / 3
        """
        p = (high + low + close) / 3.0
        r1 = (2.0 * p) - low
        s1 = (2.0 * p) - high
        r2 = p + (high - low)
        s2 = p - (high - low)
        r3 = high + 2.0 * (p - low)
        s3 = low - 2.0 * (high - p)
        r4 = r3 + (high - low)
        s4 = s3 - (high - low)

        return {
            "p": round(p, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "r4": round(r4, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
            "s4": round(s4, 2),
        }

    @staticmethod
    def calculate_fibonacci_pivots(
        high: float, low: float, close: float
    ) -> Dict[str, float]:
        """
        Fibonacci Pivot Points.
        Uses 0.382, 0.618, and 1.000 multiples of the prior range.
        """
        p = (high + low + close) / 3.0
        rng = high - low

        r1 = p + (0.382 * rng)
        s1 = p - (0.382 * rng)
        r2 = p + (0.618 * rng)
        s2 = p - (0.618 * rng)
        r3 = p + (1.000 * rng)
        s3 = p - (1.000 * rng)

        return {
            "p": round(p, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
        }

    @staticmethod
    def calculate_camarilla_pivots(
        high: float, low: float, close: float
    ) -> Dict[str, float]:
        """
        Camarilla Pivot Points.
        R3/S3 represent key mean-reversion reversal levels.
        R4/S4 represent key breakout / breakdown acceleration levels.
        """
        p = (high + low + close) / 3.0
        rng = high - low

        r4 = close + (rng * 1.1 / 2.0)
        s4 = close - (rng * 1.1 / 2.0)
        r3 = close + (rng * 1.1 / 4.0)
        s3 = close - (rng * 1.1 / 4.0)
        r2 = close + (rng * 1.1 / 6.0)
        s2 = close - (rng * 1.1 / 6.0)
        r1 = close + (rng * 1.1 / 12.0)
        s1 = close - (rng * 1.1 / 12.0)

        return {
            "p": round(p, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "r4": round(r4, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
            "s4": round(s4, 2),
        }

    @staticmethod
    def calculate_woodie_pivots(
        high: float, low: float, close: float, open_p: float
    ) -> Dict[str, float]:
        """
        Woodie Pivot Points.
        P = (H + L + 2 * Open) / 4 (weights the opening price).
        """
        p = (high + low + (2.0 * open_p)) / 4.0
        rng = high - low

        r1 = (2.0 * p) - low
        s1 = (2.0 * p) - high
        r2 = p + rng
        s2 = p - rng
        r3 = high + 2.0 * (p - low)
        s3 = low - 2.0 * (high - p)

        return {
            "p": round(p, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
        }

    def calculate_all_pivots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all four pivot point variations using the prior trading candle.
        Identifies current price position relative to central pivot and nearest levels.

        Returns:
            Dict containing Standard, Fibonacci, Camarilla, and Woodie levels.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 2:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Need at least 2 candles for pivot calculation",
            }

        # Reference prior completed bar (or previous day)
        prior_bar = clean.iloc[-2]
        curr_bar = clean.iloc[-1]

        h = float(prior_bar["High"])
        l = float(prior_bar["Low"])
        c = float(prior_bar["Close"])
        o_curr = float(curr_bar["Open"]) if "Open" in curr_bar else c
        current_price = float(curr_bar["Close"])

        standard = self.calculate_standard_pivots(h, l, c)
        fib = self.calculate_fibonacci_pivots(h, l, c)
        camarilla = self.calculate_camarilla_pivots(h, l, c)
        woodie = self.calculate_woodie_pivots(h, l, c, o_curr)

        # Find nearest support and resistance across Standard pivots
        p_std = standard["p"]
        if current_price >= p_std:
            signal = "BUY"
            bias = "ABOVE_CENTRAL_PIVOT"
            confidence = 68.0
        else:
            signal = "SELL"
            bias = "BELOW_CENTRAL_PIVOT"
            confidence = 68.0

        all_std_levels = [
            ("R4", standard["r4"]),
            ("R3", standard["r3"]),
            ("R2", standard["r2"]),
            ("R1", standard["r1"]),
            ("P", standard["p"]),
            ("S1", standard["s1"]),
            ("S2", standard["s2"]),
            ("S3", standard["s3"]),
            ("S4", standard["s4"]),
        ]

        supports = [lvl for lvl in all_std_levels if lvl[1] < current_price]
        resistances = [lvl for lvl in all_std_levels if lvl[1] > current_price]

        nearest_support = max(supports, key=lambda x: x[1]) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x[1]) if resistances else None

        return {
            "value": {
                "current_price": round(current_price, 2),
                "standard": standard,
                "fibonacci": fib,
                "camarilla": camarilla,
                "woodie": woodie,
            },
            "signal": signal,
            "confidence": confidence,
            "bias": bias,
            "nearest_standard_support": {
                "name": nearest_support[0] if nearest_support else "None",
                "price": nearest_support[1] if nearest_support else None,
            },
            "nearest_standard_resistance": {
                "name": nearest_resistance[0] if nearest_resistance else "None",
                "price": nearest_resistance[1] if nearest_resistance else None,
            },
        }

    # ==========================================
    # 2. FIBONACCI RETRACEMENT
    # ==========================================

    def calculate_fibonacci_retracement(
        self, df: pd.DataFrame, lookback: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate Fibonacci Retracement levels from the most prominent swing high and swing low.
        Levels: 23.6%, 38.2%, 50.0%, 61.8% (Golden Pocket), 78.6%.

        Returns:
            Dict containing retracement levels, swing anchors, current test status,
            BUY/SELL/NEUTRAL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Fibonacci Retracement",
            }

        sample = clean.tail(min(len(clean), lookback))
        high_idx = sample["High"].idxmax()
        low_idx = sample["Low"].idxmin()

        swing_high = float(sample.loc[high_idx, "High"])
        swing_low = float(sample.loc[low_idx, "Low"])
        current_price = float(clean["Close"].iloc[-1])

        diff = swing_high - swing_low
        if diff <= 0:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Flat price range",
            }

        # Trend direction based on chronological order of swings
        pos_high = sample.index.get_loc(high_idx)
        pos_low = sample.index.get_loc(low_idx)

        is_uptrend = pos_low < pos_high  # Low formed first, then high

        # Standard Fibonacci Retracement Ratios
        ratios = [
            ("0.0%", 0.000, 40),
            ("23.6%", 0.236, 65),
            ("38.2%", 0.382, 80),
            ("50.0%", 0.500, 75),
            ("61.8%", 0.618, 92),  # Golden Pocket
            ("78.6%", 0.786, 75),
            ("100.0%", 1.000, 85),
        ]

        levels: List[Dict[str, Any]] = []

        for name, r, strength in ratios:
            if is_uptrend:
                # Retracing downwards from the peak
                level_price = swing_high - (diff * r)
            else:
                # Retracing upwards from the trough
                level_price = swing_low + (diff * r)

            levels.append({
                "level": name,
                "ratio": r,
                "price": round(level_price, 2),
                "strength": strength,
                "is_golden_pocket": name == "61.8%",
            })

        # Test if current price is testing any Fibonacci level (within 0.8%)
        active_test = None
        closest_level = min(levels, key=lambda x: abs(x["price"] - current_price))
        dist_pct = abs(closest_level["price"] - current_price) / current_price * 100.0

        if dist_pct <= 0.8:
            active_test = {
                "level": closest_level["level"],
                "price": closest_level["price"],
                "distance_pct": round(dist_pct, 2),
            }

        # Signal determination
        if is_uptrend:
            # In uptrend, pullbacks to 50% or 61.8% are high-probability BUY zones
            if active_test and active_test["level"] in ("50.0%", "61.8%"):
                signal = "BUY"
                confidence = 88.0
                status = "TESTING_GOLDEN_POCKET_SUPPORT"
            elif current_price > (swing_high - diff * 0.382):
                signal = "BUY"
                confidence = 74.0
                status = "SHALLOW_PULLBACK_STRONG_UPTREND"
            elif current_price < (swing_high - diff * 0.786):
                signal = "SELL"
                confidence = 72.0
                status = "DEEP_PULLBACK_RISK_OF_REVERSAL"
            else:
                signal = "BUY"
                confidence = 66.0
                status = "UPTREND_HEALTHY_RETRACEMENT"
        else:
            # In downtrend, bounces to 50% or 61.8% are high-probability SELL zones
            if active_test and active_test["level"] in ("50.0%", "61.8%"):
                signal = "SELL"
                confidence = 88.0
                status = "TESTING_GOLDEN_POCKET_RESISTANCE"
            elif current_price < (swing_low + diff * 0.382):
                signal = "SELL"
                confidence = 74.0
                status = "SHALLOW_BOUNCE_STRONG_DOWNTREND"
            elif current_price > (swing_low + diff * 0.786):
                signal = "BUY"
                confidence = 72.0
                status = "DEEP_BOUNCE_RISK_OF_REVERSAL"
            else:
                signal = "SELL"
                confidence = 66.0
                status = "DOWNTREND_HEALTHY_RETRACEMENT"

        return {
            "value": {
                "swing_high": round(swing_high, 2),
                "swing_low": round(swing_low, 2),
                "current_price": round(current_price, 2),
                "trend": "UPTREND" if is_uptrend else "DOWNTREND",
                "levels": levels,
            },
            "signal": signal,
            "confidence": confidence,
            "status": status,
            "active_test": active_test,
        }

    # ==========================================
    # 3. FIBONACCI EXTENSION
    # ==========================================

    def calculate_fibonacci_extensions(
        self, df: pd.DataFrame, lookback: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate Fibonacci Extension levels (100%, 127.2%, 161.8%, 200%, 261.8%)
        for projection targets after breakout / continuation.

        Returns:
            Dict containing target extension levels, distances, BUY/SELL signal, and confidence.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Fibonacci Extensions",
            }

        sample = clean.tail(min(len(clean), lookback))
        high_idx = sample["High"].idxmax()
        low_idx = sample["Low"].idxmin()

        swing_high = float(sample.loc[high_idx, "High"])
        swing_low = float(sample.loc[low_idx, "Low"])
        current_price = float(clean["Close"].iloc[-1])
        diff = swing_high - swing_low

        if diff <= 0:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Invalid price range",
            }

        pos_high = sample.index.get_loc(high_idx)
        pos_low = sample.index.get_loc(low_idx)
        is_uptrend = pos_low < pos_high

        # Extension ratios and historical probability weights
        extension_ratios = [
            ("100.0%", 1.000, 75),
            ("127.2%", 1.272, 85),
            ("161.8%", 1.618, 92),  # Golden Ratio Extension
            ("200.0%", 2.000, 70),
            ("261.8%", 2.618, 80),
        ]

        extensions: List[Dict[str, Any]] = []

        for name, r, strength in extension_ratios:
            if is_uptrend:
                target_price = swing_low + (diff * r)
                dist_pct = ((target_price - current_price) / current_price) * 100.0
            else:
                target_price = swing_high - (diff * r)
                dist_pct = ((current_price - target_price) / current_price) * 100.0

            extensions.append({
                "level": name,
                "ratio": r,
                "price": round(target_price, 2),
                "distance_pct": round(dist_pct, 2),
                "strength": strength,
            })

        signal = "BUY" if is_uptrend else "SELL"
        confidence = 75.0

        return {
            "value": {
                "trend": "UPTREND_TARGETS" if is_uptrend else "DOWNTREND_TARGETS",
                "current_price": round(current_price, 2),
                "extensions": extensions,
            },
            "signal": signal,
            "confidence": confidence,
        }

    # ==========================================
    # 4. DYNAMIC S/R FROM PRICE ACTION (FRACTALS)
    # ==========================================

    def calculate_dynamic_sr(
        self, df: pd.DataFrame, window: int = 5, cluster_pct: float = 0.012
    ) -> Dict[str, Any]:
        """
        Detect dynamic horizontal support and resistance zones using
        Bill Williams 5-bar swing fractals and clustering.

        Args:
            df: OHLCV DataFrame
            window: Number of neighboring bars for fractal peak/trough detection
            cluster_pct: Percentage threshold (default 1.2%) to group touches into zones

        Returns:
            Dict containing sorted support_zones and resistance_zones with touch counts and strengths.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < (window * 2 + 5):
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Fractal S/R",
            }

        highs = clean["High"].values
        lows = clean["Low"].values
        vols = clean["Volume"].values if "Volume" in clean else np.ones(len(clean))
        dates = clean.index
        n = len(clean)

        fractal_highs = []
        fractal_lows = []

        # 5-bar fractal detection (2 bars before, 2 bars after)
        half_w = window // 2
        for i in range(half_w, n - half_w):
            # Swing High
            if highs[i] == max(highs[i - half_w : i + half_w + 1]):
                fractal_highs.append({
                    "price": float(highs[i]),
                    "date": dates[i],
                    "volume": float(vols[i]),
                    "age": n - 1 - i,
                })
            # Swing Low
            if lows[i] == min(lows[i - half_w : i + half_w + 1]):
                fractal_lows.append({
                    "price": float(lows[i]),
                    "date": dates[i],
                    "volume": float(vols[i]),
                    "age": n - 1 - i,
                })

        current_price = float(clean["Close"].iloc[-1])

        # Cluster points into zones
        def cluster_fractals(points: List[Dict[str, Any]], zone_type: str) -> List[Dict[str, Any]]:
            if not points:
                return []

            points_sorted = sorted(points, key=lambda x: x["price"])
            clusters: List[List[Dict[str, Any]]] = []

            for p in points_sorted:
                if not clusters:
                    clusters.append([p])
                else:
                    cluster_mean = np.mean([pt["price"] for pt in clusters[-1]])
                    if abs(p["price"] - cluster_mean) / cluster_mean <= cluster_pct:
                        clusters[-1].append(p)
                    else:
                        clusters.append([p])

            zones = []
            for cl in clusters:
                cluster_prices = [pt["price"] for pt in cl]
                cluster_vols = [pt["volume"] for pt in cl]
                cluster_ages = [pt["age"] for pt in cl]

                touch_count = len(cl)
                weighted_price = float(np.average(cluster_prices, weights=cluster_vols if sum(cluster_vols) > 0 else None))
                min_price = min(cluster_prices)
                max_price = max(cluster_prices)
                min_age = min(cluster_ages)  # most recent touch

                # Strength calculation (0-100)
                # Base touches: 1=40, 2=65, 3=80, 4+=92
                if touch_count == 1:
                    base_str = 40.0
                elif touch_count == 2:
                    base_str = 65.0
                elif touch_count == 3:
                    base_str = 82.0
                else:
                    base_str = 94.0

                # Recency bonus (within last 30 bars)
                recency_bonus = max(0.0, (60.0 - min_age) * 0.15) if min_age <= 60 else 0.0
                final_strength = min(round(base_str + recency_bonus, 1), 98.0)

                dist_pct = ((weighted_price - current_price) / current_price) * 100.0

                zones.append({
                    "price": round(weighted_price, 2),
                    "range_low": round(min_price, 2),
                    "range_high": round(max_price, 2),
                    "touch_count": touch_count,
                    "strength": final_strength,
                    "type": zone_type,
                    "distance_pct": round(dist_pct, 2),
                    "most_recent_touch_bars_ago": int(min_age),
                })

            return zones

        high_zones = cluster_fractals(fractal_highs, "RESISTANCE")
        low_zones = cluster_fractals(fractal_lows, "SUPPORT")

        # Separate zones into supports (< current price) and resistances (> current price)
        all_zones = high_zones + low_zones
        supports = [z for z in all_zones if z["price"] < current_price]
        resistances = [z for z in all_zones if z["price"] > current_price]

        # Sort supports descending (closest first), resistances ascending (closest first)
        supports = sorted(supports, key=lambda x: x["price"], reverse=True)
        resistances = sorted(resistances, key=lambda x: x["price"])

        # Determine signal based on proximity to nearest support vs resistance
        nearest_sup = supports[0] if supports else None
        nearest_res = resistances[0] if resistances else None

        if nearest_sup and abs(nearest_sup["distance_pct"]) <= 1.0:
            signal = "BUY"
            confidence = nearest_sup["strength"]
            status = "AT_MAJOR_SUPPORT_ZONE"
        elif nearest_res and abs(nearest_res["distance_pct"]) <= 1.0:
            signal = "SELL"
            confidence = nearest_res["strength"]
            status = "AT_MAJOR_RESISTANCE_ZONE"
        elif nearest_sup and nearest_res:
            dist_to_sup = abs(nearest_sup["distance_pct"])
            dist_to_res = abs(nearest_res["distance_pct"])
            if dist_to_res > dist_to_sup * 1.5:
                signal = "BUY"
                confidence = 68.0
                status = "FAVORABLE_RISK_REWARD_NEAR_SUPPORT"
            elif dist_to_sup > dist_to_res * 1.5:
                signal = "SELL"
                confidence = 68.0
                status = "UNFAVORABLE_RISK_REWARD_NEAR_RESISTANCE"
            else:
                signal = "NEUTRAL"
                confidence = 52.0
                status = "MID_RANGE"
        else:
            signal = "NEUTRAL"
            confidence = 50.0
            status = "NEUTRAL"

        return {
            "value": {
                "current_price": round(current_price, 2),
                "support_zones": supports[:5],
                "resistance_zones": resistances[:5],
                "total_fractals_detected": len(fractal_highs) + len(fractal_lows),
            },
            "signal": signal,
            "confidence": confidence,
            "status": status,
        }

    # ==========================================
    # 5. CONFLUENCE SCORING (<= 1% CLUSTERING)
    # ==========================================

    def calculate_confluence_zones(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        cluster_pct: float = 0.010,
    ) -> Dict[str, Any]:
        """
        Aggregate all structural levels across Pivots, Fibonacci Retracement,
        Extensions, Dynamic Price Action, Moving Averages, and Volume Profile.
        Groups levels within 1.0% tolerance into high-probability Confluence Zones.

        Args:
            df: OHLCV DataFrame
            current_price: Optional current price override
            cluster_pct: Clustering tolerance (default 1.0%)

        Returns:
            Dict containing ranked confluence zones, primary support, primary resistance,
            risk/reward evaluation, and confidence scores.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "Insufficient data for Confluence Scoring",
            }

        if current_price is None:
            current_price = float(clean["Close"].iloc[-1])

        collected_levels: List[Dict[str, Any]] = []

        # 1. Pivot levels (Standard & Fibonacci)
        pivots_data = self.calculate_all_pivots(clean)
        if "value" in pivots_data and pivots_data["value"]:
            std = pivots_data["value"].get("standard", {})
            for k in ["r1", "r2", "r3", "s1", "s2", "s3", "p"]:
                if k in std:
                    collected_levels.append({
                        "name": f"Classic Pivot {k.upper()}",
                        "price": std[k],
                        "source": "PIVOT_CLASSIC",
                    })

            fib_p = pivots_data["value"].get("fibonacci", {})
            for k in ["r1", "r2", "r3", "s1", "s2", "s3"]:
                if k in fib_p:
                    collected_levels.append({
                        "name": f"Fibonacci Pivot {k.upper()}",
                        "price": fib_p[k],
                        "source": "PIVOT_FIBONACCI",
                    })

            cam = pivots_data["value"].get("camarilla", {})
            for k in ["r3", "r4", "s3", "s4"]:
                if k in cam:
                    collected_levels.append({
                        "name": f"Camarilla {k.upper()}",
                        "price": cam[k],
                        "source": "PIVOT_CAMARILLA",
                    })

        # 2. Fibonacci Retracements
        fib_ret = self.calculate_fibonacci_retracement(clean)
        if "value" in fib_ret and "levels" in fib_ret["value"]:
            for item in fib_ret["value"]["levels"]:
                collected_levels.append({
                    "name": f"Fib Retracement {item['level']}",
                    "price": item["price"],
                    "source": "FIBONACCI_RETRACEMENT",
                })

        # 3. Fibonacci Extensions
        fib_ext = self.calculate_fibonacci_extensions(clean)
        if "value" in fib_ext and "extensions" in fib_ext["value"]:
            for item in fib_ext["value"]["extensions"][:3]:
                collected_levels.append({
                    "name": f"Fib Extension {item['level']}",
                    "price": item["price"],
                    "source": "FIBONACCI_EXTENSION",
                })

        # 4. Dynamic S/R zones
        dyn_sr = self.calculate_dynamic_sr(clean)
        if "value" in dyn_sr:
            for s in dyn_sr["value"].get("support_zones", [])[:3]:
                collected_levels.append({
                    "name": f"Fractal Support ({s['touch_count']} touches)",
                    "price": s["price"],
                    "source": "FRACTAL_SUPPORT",
                })
            for r in dyn_sr["value"].get("resistance_zones", [])[:3]:
                collected_levels.append({
                    "name": f"Fractal Resistance ({r['touch_count']} touches)",
                    "price": r["price"],
                    "source": "FRACTAL_RESISTANCE",
                })

        # 5. Major Moving Averages (EMA 20, 50, 100, 200)
        close_series = clean["Close"]
        for p in [20, 50, 100, 200]:
            if len(clean) >= p:
                ema_val = ta.ema(close_series, length=p)
                if ema_val is not None and not pd.isna(ema_val.iloc[-1]):
                    collected_levels.append({
                        "name": f"EMA {p}",
                        "price": round(float(ema_val.iloc[-1]), 2),
                        "source": "MOVING_AVERAGE",
                    })

        if not collected_levels:
            return {
                "value": {},
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "error": "No levels collected",
            }

        # Cluster all levels within cluster_pct (1.0%)
        sorted_lvls = sorted(collected_levels, key=lambda x: x["price"])
        clusters: List[List[Dict[str, Any]]] = []

        for lvl in sorted_lvls:
            if not clusters:
                clusters.append([lvl])
            else:
                cluster_avg = np.mean([x["price"] for x in clusters[-1]])
                if abs(lvl["price"] - cluster_avg) / cluster_avg <= cluster_pct:
                    clusters[-1].append(lvl)
                else:
                    clusters.append([lvl])

        confluence_zones: List[Dict[str, Any]] = []

        for cl in clusters:
            prices = [x["price"] for x in cl]
            names = [x["name"] for x in cl]
            sources = list(set([x["source"] for x in cl]))

            avg_price = float(np.mean(prices))
            min_price = min(prices)
            max_price = max(prices)
            level_count = len(cl)
            unique_sources_count = len(sources)

            # Confluence strength score (0-100)
            # 1 level = 35, 2 levels = 58, 3 levels = 76, 4 levels = 88, 5+ = 96
            if unique_sources_count == 1:
                strength = 35.0
            elif unique_sources_count == 2:
                strength = 58.0
            elif unique_sources_count == 3:
                strength = 76.0
            elif unique_sources_count == 4:
                strength = 88.0
            else:
                strength = 96.0

            dist_pct = ((avg_price - current_price) / current_price) * 100.0

            if abs(dist_pct) <= 0.5:
                zone_type = "CURRENT_TEST_ZONE"
            elif avg_price < current_price:
                zone_type = "SUPPORT"
            else:
                zone_type = "RESISTANCE"

            confluence_zones.append({
                "price": round(avg_price, 2),
                "range_low": round(min_price, 2),
                "range_high": round(max_price, 2),
                "level_count": level_count,
                "unique_sources_count": unique_sources_count,
                "strength": strength,
                "type": zone_type,
                "distance_pct": round(dist_pct, 2),
                "contributing_levels": names,
            })

        # Rank zones: primary support and primary resistance
        supports = [z for z in confluence_zones if z["type"] == "SUPPORT"]
        resistances = [z for z in confluence_zones if z["type"] == "RESISTANCE"]

        # Primary support: highest strength, or closest if equal
        primary_sup = max(supports, key=lambda x: (x["strength"], x["price"])) if supports else None
        primary_res = max(resistances, key=lambda x: (x["strength"], -x["price"])) if resistances else None

        # Nearest support and resistance
        nearest_sup = max(supports, key=lambda x: x["price"]) if supports else None
        nearest_res = min(resistances, key=lambda x: x["price"]) if resistances else None

        # Risk-to-reward ratio
        if nearest_sup and nearest_res:
            downside = abs(nearest_sup["distance_pct"])
            upside = abs(nearest_res["distance_pct"])
            rr_ratio = round(upside / downside, 2) if downside > 0 else 1.0
        else:
            rr_ratio = 1.0

        # Overall signal
        if nearest_sup and abs(nearest_sup["distance_pct"]) <= 0.8:
            signal = "BUY"
            confidence = nearest_sup["strength"]
        elif nearest_res and abs(nearest_res["distance_pct"]) <= 0.8:
            signal = "SELL"
            confidence = nearest_res["strength"]
        elif rr_ratio >= 2.0:
            signal = "BUY"
            confidence = 72.0
        elif rr_ratio <= 0.5:
            signal = "SELL"
            confidence = 72.0
        else:
            signal = "NEUTRAL"
            confidence = 50.0

        return {
            "value": {
                "current_price": round(current_price, 2),
                "primary_support": primary_sup,
                "primary_resistance": primary_res,
                "nearest_support": nearest_sup,
                "nearest_resistance": nearest_res,
                "risk_reward_ratio": rr_ratio,
                "confluence_zones": sorted(confluence_zones, key=lambda x: x["strength"], reverse=True)[:8],
            },
            "signal": signal,
            "confidence": confidence,
        }

    # ==========================================
    # 6. MASTER ANALYZE METHODS
    # ==========================================

    def analyze(
        self, df: pd.DataFrame, current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute full Support & Resistance analysis suite on DataFrame.

        Args:
            df: OHLCV DataFrame
            current_price: Optional price override

        Returns:
            Structured dictionary with Pivots, Fibonacci, Dynamic S/R, Confluence,
            and overall Summary.
        """
        clean = self._clean_df(df)
        if clean.empty or len(clean) < 15:
            return {
                "current_price": None,
                "error": "Insufficient data for support & resistance analysis",
                "disclaimer": self.disclaimer,
            }

        if current_price is None:
            current_price = float(clean["Close"].iloc[-1])

        pivots = self.calculate_all_pivots(clean)
        fib_ret = self.calculate_fibonacci_retracement(clean)
        fib_ext = self.calculate_fibonacci_extensions(clean)
        dynamic_sr = self.calculate_dynamic_sr(clean)
        confluence = self.calculate_confluence_zones(clean, current_price=current_price)

        conf_val = confluence.get("value", {})
        primary_sup = conf_val.get("primary_support", {}) or {}
        primary_res = conf_val.get("primary_resistance", {}) or {}
        nearest_sup = conf_val.get("nearest_support", {}) or {}
        nearest_res = conf_val.get("nearest_resistance", {}) or {}

        summary = {
            "primary_support": primary_sup.get("price"),
            "primary_support_strength": primary_sup.get("strength"),
            "primary_resistance": primary_res.get("price"),
            "primary_resistance_strength": primary_res.get("strength"),
            "nearest_support": nearest_sup.get("price"),
            "nearest_resistance": nearest_res.get("price"),
            "risk_reward_ratio": conf_val.get("risk_reward_ratio", 1.0),
            "signal": confluence.get("signal", "NEUTRAL"),
            "confidence": confluence.get("confidence", 50.0),
        }

        return {
            "current_price": round(current_price, 2),
            "pivots": pivots.get("value", {}),
            "fibonacci_retracement": fib_ret.get("value", {}),
            "fibonacci_extensions": fib_ext.get("value", {}),
            "dynamic_sr": dynamic_sr.get("value", {}),
            "confluence": conf_val,
            "summary": summary,
            "disclaimer": self.disclaimer,
        }

    def analyze_symbol(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch data for stock symbol and run complete support/resistance analysis.

        Args:
            symbol: NSE stock symbol (e.g. 'RELIANCE', 'TCS')
            period: OHLCV history lookback period

        Returns:
            Structured S/R analysis result dictionary.
        """
        clean_sym = symbol.upper().replace(".NS", "").replace(".BO", "")
        logger.info(f"Running Support & Resistance analysis for {clean_sym}")

        df = StockDataFetcher.get_ohlcv(clean_sym, period=period)
        if df.empty:
            return {
                "symbol": clean_sym,
                "current_price": None,
                "error": f"No data returned for symbol {clean_sym}",
                "disclaimer": self.disclaimer,
            }

        res = self.analyze(df)
        res["symbol"] = clean_sym
        return res
