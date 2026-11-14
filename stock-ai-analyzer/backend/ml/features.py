"""
Feature Engineering Module for ML Stock Prediction.
Generates 50+ quantitative, technical, volatility, and lag features from OHLCV data.
Follows strict quantitative finance standards with zero look-ahead bias.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

try:
    import pandas_ta as ta
except ImportError:
    ta = None

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline producing 50+ ML features for Indian equities.
    Supports Price, Technical (via pandas_ta), Volatility, and Lag features,
    along with multi-horizon classification target variables.
    """

    FEATURE_NAMES: List[str] = [
        # Price Features (12)
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "log_return_1d",
        "log_return_5d",
        "hl_range",
        "co_range",
        "dist_52w_high",
        "dist_52w_low",
        "close_to_high",
        "close_to_low",
        # Technical EMA Ratios (5)
        "close_to_ema20",
        "close_to_ema50",
        "close_to_ema200",
        "ema20_to_ema50",
        "ema50_to_ema200",
        # Technical Oscillators & Trends (15)
        "rsi_14",
        "rsi_7",
        "macd",
        "macd_signal",
        "macd_hist",
        "stoch_k",
        "stoch_d",
        "adx_14",
        "dmp_14",
        "dmn_14",
        "bbands_pct_b",
        "bbands_bandwidth",
        "atr_14",
        "atr_ratio",
        "supertrend_dir",
        # Volume & Flow Features (5)
        "volume_ratio",
        "vol_ratio_5",
        "obv_change_rate",
        "mfi_14",
        "cci_14",
        # Volatility Features (7)
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "garman_klass_vol",
        "parkinson_vol",
        "vol_ratio_5_20",
        "normalized_atr",
        # Lag Features (14)
        "lag_return_1d",
        "lag_return_2d",
        "lag_return_3d",
        "lag_return_5d",
        "lag_return_10d",
        "lag_vol_ratio_1d",
        "lag_vol_ratio_2d",
        "lag_vol_ratio_3d",
        "lag_rsi_1d",
        "lag_rsi_3d",
        "lag_rsi_5d",
        "lag_macd_hist_1d",
        "lag_macd_hist_2d",
        "lag_macd_hist_3d",
    ]

    def __init__(self):
        """Initialize FeatureEngineer."""
        self._feature_names = list(self.FEATURE_NAMES)

    def get_feature_names(self) -> List[str]:
        """
        Return list of generated feature column names.

        Returns:
            List of 58 feature column names.
        """
        return list(self._feature_names)

    @staticmethod
    def _extract_ohlcv(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Extract Open, High, Low, Close, Volume series agnostic of column casing.
        """
        col_map = {str(c).lower().strip(): c for c in df.columns}

        def get_col(candidates: List[str], default_name: str) -> pd.Series:
            for cand in candidates:
                if cand in col_map:
                    return df[col_map[cand]].astype(float)
            if default_name in df.columns:
                return df[default_name].astype(float)
            raise ValueError(f"Required OHLCV column matching {candidates} not found in {list(df.columns)}")

        open_s = get_col(["open"], "Open")
        high_s = get_col(["high"], "High")
        low_s = get_col(["low"], "Low")
        close_s = get_col(["close", "adj close", "adj_close"], "Close")
        volume_s = get_col(["volume", "vol"], "Volume")

        return open_s, high_s, low_s, close_s, volume_s

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 50+ quantitative, technical, volatility, and lag features on OHLCV data.

        Args:
            df: DataFrame containing Open, High, Low, Close, Volume.

        Returns:
            DataFrame with all feature columns added.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame()

        data = df.copy()
        open_s, high_s, low_s, close_s, volume_s = self._extract_ohlcv(data)

        # ---------------------------------------------------------
        # 1. Price Features
        # ---------------------------------------------------------
        safe_close = close_s.replace(0, np.nan)
        safe_open = open_s.replace(0, np.nan)

        data["return_1d"] = safe_close.pct_change(1)
        data["return_5d"] = safe_close.pct_change(5)
        data["return_10d"] = safe_close.pct_change(10)
        data["return_20d"] = safe_close.pct_change(20)

        data["log_return_1d"] = np.log(safe_close / safe_close.shift(1).replace(0, np.nan))
        data["log_return_5d"] = np.log(safe_close / safe_close.shift(5).replace(0, np.nan))

        data["hl_range"] = (high_s - low_s) / safe_close
        data["co_range"] = (close_s - open_s) / safe_close

        roll_52w_high = high_s.rolling(252, min_periods=20).max()
        roll_52w_low = low_s.rolling(252, min_periods=20).min()
        data["dist_52w_high"] = (close_s - roll_52w_high) / roll_52w_high.replace(0, np.nan)
        data["dist_52w_low"] = (close_s - roll_52w_low) / roll_52w_low.replace(0, np.nan)

        data["close_to_high"] = (high_s - close_s) / safe_close
        data["close_to_low"] = (close_s - low_s) / safe_close

        # ---------------------------------------------------------
        # 2. Technical Features (pandas_ta with analytical fallbacks)
        # ---------------------------------------------------------
        # EMAs and EMA ratios
        ema20 = close_s.ewm(span=20, adjust=False).mean()
        ema50 = close_s.ewm(span=50, adjust=False).mean()
        ema200 = close_s.ewm(span=200, adjust=False).mean()

        data["close_to_ema20"] = close_s / ema20.replace(0, np.nan)
        data["close_to_ema50"] = close_s / ema50.replace(0, np.nan)
        data["close_to_ema200"] = close_s / ema200.replace(0, np.nan)
        data["ema20_to_ema50"] = ema20 / ema50.replace(0, np.nan)
        data["ema50_to_ema200"] = ema50 / ema200.replace(0, np.nan)

        # RSI 14 & 7
        data["rsi_14"] = self._compute_rsi(close_s, period=14)
        data["rsi_7"] = self._compute_rsi(close_s, period=7)

        # MACD
        macd_line, macd_sig, macd_h = self._compute_macd(close_s)
        data["macd"] = macd_line
        data["macd_signal"] = macd_sig
        data["macd_hist"] = macd_h

        # Stochastic (14, 3, 3)
        stoch_k, stoch_d = self._compute_stochastic(high_s, low_s, close_s)
        data["stoch_k"] = stoch_k
        data["stoch_d"] = stoch_d

        # ADX (14)
        adx, dmp, dmn = self._compute_adx(high_s, low_s, close_s)
        data["adx_14"] = adx
        data["dmp_14"] = dmp
        data["dmn_14"] = dmn

        # Bollinger Bands (20, 2)
        bb_pct_b, bb_bw = self._compute_bollinger(close_s)
        data["bbands_pct_b"] = bb_pct_b
        data["bbands_bandwidth"] = bb_bw

        # ATR & ATR ratio
        atr_14 = self._compute_atr(high_s, low_s, close_s, period=14)
        data["atr_14"] = atr_14
        data["atr_ratio"] = atr_14 / safe_close

        # Supertrend direction (1 = uptrend, -1 = downtrend)
        data["supertrend_dir"] = self._compute_supertrend_dir(high_s, low_s, close_s)

        # ---------------------------------------------------------
        # 3. Volume & Flow Features
        # ---------------------------------------------------------
        vol_sma20 = volume_s.rolling(20, min_periods=5).mean()
        vol_sma5 = volume_s.rolling(5, min_periods=2).mean()
        data["volume_ratio"] = volume_s / vol_sma20.replace(0, np.nan)
        data["vol_ratio_5"] = volume_s / vol_sma5.replace(0, np.nan)

        obv = self._compute_obv(close_s, volume_s)
        data["obv_change_rate"] = obv.pct_change(5).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        data["mfi_14"] = self._compute_mfi(high_s, low_s, close_s, volume_s, period=14)
        data["cci_14"] = self._compute_cci(high_s, low_s, close_s, period=14)

        # ---------------------------------------------------------
        # 4. Volatility Features
        # ---------------------------------------------------------
        ret1 = data["return_1d"]
        data["vol_5d"] = ret1.rolling(5, min_periods=3).std() * np.sqrt(252)
        data["vol_10d"] = ret1.rolling(10, min_periods=5).std() * np.sqrt(252)
        data["vol_20d"] = ret1.rolling(20, min_periods=10).std() * np.sqrt(252)

        data["garman_klass_vol"] = self._compute_garman_klass_vol(high_s, low_s, open_s, close_s)
        data["parkinson_vol"] = self._compute_parkinson_vol(high_s, low_s)
        data["vol_ratio_5_20"] = data["vol_5d"] / data["vol_20d"].replace(0, np.nan)
        data["normalized_atr"] = atr_14 / safe_close.rolling(20, min_periods=5).mean().replace(0, np.nan)

        # ---------------------------------------------------------
        # 5. Lag Features
        # ---------------------------------------------------------
        data["lag_return_1d"] = data["return_1d"].shift(1)
        data["lag_return_2d"] = data["return_1d"].shift(2)
        data["lag_return_3d"] = data["return_1d"].shift(3)
        data["lag_return_5d"] = data["return_1d"].shift(5)
        data["lag_return_10d"] = data["return_1d"].shift(10)

        data["lag_vol_ratio_1d"] = data["volume_ratio"].shift(1)
        data["lag_vol_ratio_2d"] = data["volume_ratio"].shift(2)
        data["lag_vol_ratio_3d"] = data["volume_ratio"].shift(3)

        data["lag_rsi_1d"] = data["rsi_14"].shift(1)
        data["lag_rsi_3d"] = data["rsi_14"].shift(3)
        data["lag_rsi_5d"] = data["rsi_14"].shift(5)

        data["lag_macd_hist_1d"] = data["macd_hist"].shift(1)
        data["lag_macd_hist_2d"] = data["macd_hist"].shift(2)
        data["lag_macd_hist_3d"] = data["macd_hist"].shift(3)

        # Clean infinities
        for feat in self.FEATURE_NAMES:
            if feat in data.columns:
                data[feat] = data[feat].replace([np.inf, -np.inf], np.nan)

        return data

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        horizon: int = 5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepares training-ready feature matrix X and directional binary target y.
        Drops look-ahead and unpopulated NaN rows.

        Args:
            df: Historical OHLCV DataFrame.
            horizon: Forward horizon in trading days (1=daily, 5=weekly, 20=monthly).

        Returns:
            Tuple of (X: pd.DataFrame, y: pd.Series)
        """
        if df is None or len(df) <= horizon + 20:
            return pd.DataFrame(), pd.Series(dtype=int)

        # Check if features are already present
        feat_set = set(self.FEATURE_NAMES)
        if not feat_set.issubset(df.columns):
            featured_df = self.compute_features(df)
        else:
            featured_df = df.copy()

        _, _, _, close_s, _ = self._extract_ohlcv(featured_df)

        # Target: 1 if forward return > 0 else 0
        forward_return = (close_s.shift(-horizon) - close_s) / close_s.replace(0, np.nan)
        target = (forward_return > 0.0).astype(int)

        # Extract only feature columns
        X = featured_df[self.FEATURE_NAMES].copy()
        y = target.copy()

        # Target is NaN for the last `horizon` rows (unobserved future)
        valid_mask = ~(X.isna().any(axis=1) | forward_return.isna())

        X_clean = X.loc[valid_mask].astype(float)
        y_clean = y.loc[valid_mask].astype(int)

        return X_clean, y_clean

    # -------------------------------------------------------------
    # Indicator Calculation Helpers (pandas_ta + robust fallbacks)
    # -------------------------------------------------------------
    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI with pandas_ta or Wilder's smoothing fallback."""
        if ta is not None:
            try:
                res = ta.rsi(close, length=period)
                if res is not None and not res.isna().all():
                    return res
            except Exception:
                pass

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _compute_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute MACD, Signal, and Histogram."""
        if ta is not None:
            try:
                res = ta.macd(close, fast=12, slow=26, signal=9)
                if res is not None and not res.empty:
                    cols = res.columns
                    return res[cols[0]], res[cols[1]], res[cols[2]]
            except Exception:
                pass

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    @staticmethod
    def _compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Compute Stochastic %K and %D."""
        if ta is not None:
            try:
                res = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
                if res is not None and not res.empty:
                    cols = res.columns
                    return res[cols[0]], res[cols[1]]
            except Exception:
                pass

        low14 = low.rolling(14, min_periods=5).min()
        high14 = high.rolling(14, min_periods=5).max()
        denom = (high14 - low14).replace(0, np.nan)
        raw_k = 100.0 * ((close - low14) / denom)
        stoch_k = raw_k.rolling(3, min_periods=1).mean()
        stoch_d = stoch_k.rolling(3, min_periods=1).mean()
        return stoch_k.fillna(50.0), stoch_d.fillna(50.0)

    @staticmethod
    def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute ADX, +DI, -DI."""
        if ta is not None:
            try:
                res = ta.adx(high, low, close, length=period)
                if res is not None and not res.empty:
                    # columns typically [ADX_14, DMP_14, DMN_14]
                    return res.iloc[:, 0], res.iloc[:, 1], res.iloc[:, 2]
            except Exception:
                pass

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().replace(0, np.nan)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        smooth_pdm = pd.Series(plus_dm, index=high.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        smooth_mdm = pd.Series(minus_dm, index=high.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        dmp = 100.0 * (smooth_pdm / atr)
        dmn = 100.0 * (smooth_mdm / atr)
        dx = 100.0 * ((dmp - dmn).abs() / (dmp + dmn).replace(0, np.nan))
        adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        return adx.fillna(20.0), dmp.fillna(20.0), dmn.fillna(20.0)

    @staticmethod
    def _compute_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Compute Bollinger %B and Bandwidth."""
        if ta is not None:
            try:
                res = ta.bbands(close, length=period, std=num_std)
                if res is not None and not res.empty:
                    # columns typically: [BBL, BBM, BBU, BBB, BBP]
                    # return %B and Bandwidth
                    return res.iloc[:, 4], res.iloc[:, 3]
            except Exception:
                pass

        middle = close.rolling(period, min_periods=5).mean()
        std = close.rolling(period, min_periods=5).std()
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        span = (upper - lower).replace(0, np.nan)
        pct_b = (close - lower) / span
        bandwidth = span / middle.replace(0, np.nan)
        return pct_b.fillna(0.5), bandwidth.fillna(0.0)

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        if ta is not None:
            try:
                res = ta.atr(high, low, close, length=period)
                if res is not None and not res.isna().all():
                    return res
            except Exception:
                pass

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _compute_supertrend_dir(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 3.0) -> pd.Series:
        """Compute Supertrend direction (1 = Bullish, -1 = Bearish)."""
        if ta is not None:
            try:
                st = ta.supertrend(high, low, close, length=length, multiplier=multiplier)
                if st is not None and not st.empty:
                    # Column with 'd' holds direction (1 or -1)
                    dir_cols = [c for c in st.columns if "d" in c.lower()]
                    if dir_cols:
                        return st[dir_cols[0]].fillna(1).astype(float)
            except Exception:
                pass

        # Robust analytical fallback
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(length, min_periods=1).mean()

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        direction = np.ones(len(close))
        upper_band = basic_upper.values.copy()
        lower_band = basic_lower.values.copy()
        close_vals = close.values

        for i in range(1, len(close)):
            # Update upper band
            if basic_upper.values[i] < upper_band[i - 1] or close_vals[i - 1] > upper_band[i - 1]:
                upper_band[i] = basic_upper.values[i]
            else:
                upper_band[i] = upper_band[i - 1]

            # Update lower band
            if basic_lower.values[i] > lower_band[i - 1] or close_vals[i - 1] < lower_band[i - 1]:
                lower_band[i] = basic_lower.values[i]
            else:
                lower_band[i] = lower_band[i - 1]

            # Update direction
            if direction[i - 1] == 1:
                if close_vals[i] < lower_band[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
            else:
                if close_vals[i] > upper_band[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1

        return pd.Series(direction, index=close.index, dtype=float)

    @staticmethod
    def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume."""
        if ta is not None:
            try:
                res = ta.obv(close, volume)
                if res is not None:
                    return res
            except Exception:
                pass

        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Compute Money Flow Index."""
        if ta is not None:
            try:
                res = ta.mfi(high, low, close, volume, length=period)
                if res is not None:
                    return res
            except Exception:
                pass

        typical_price = (high + low + close) / 3.0
        raw_money_flow = typical_price * volume
        tp_diff = typical_price.diff()

        pos_flow = raw_money_flow.where(tp_diff > 0, 0.0)
        neg_flow = raw_money_flow.where(tp_diff < 0, 0.0)

        pos_sum = pos_flow.rolling(period, min_periods=1).sum()
        neg_sum = neg_flow.rolling(period, min_periods=1).sum().replace(0, np.nan)

        mfr = pos_sum / neg_sum
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        return mfi.fillna(50.0)

    @staticmethod
    def _compute_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Commodity Channel Index."""
        if ta is not None:
            try:
                res = ta.cci(high, low, close, length=period)
                if res is not None:
                    return res
            except Exception:
                pass

        typical_price = (high + low + close) / 3.0
        sma_tp = typical_price.rolling(period, min_periods=1).mean()
        mean_dev = (typical_price - sma_tp).abs().rolling(period, min_periods=1).mean().replace(0, np.nan)
        cci = (typical_price - sma_tp) / (0.015 * mean_dev)
        return cci.fillna(0.0)

    @staticmethod
    def _compute_garman_klass_vol(high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Compute Garman-Klass Volatility (annualized)."""
        safe_low = low.replace(0, np.nan)
        safe_open = open_.replace(0, np.nan)
        log_hl = np.log(high / safe_low)
        log_co = np.log(close / safe_open)

        term1 = 0.5 * (log_hl ** 2)
        term2 = (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        gk = (term1 - term2).clip(lower=0)

        roll_gk = gk.rolling(window, min_periods=5).mean()
        return np.sqrt(roll_gk * 252.0).fillna(0.0)

    @staticmethod
    def _compute_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """Compute Parkinson Volatility (annualized)."""
        safe_low = low.replace(0, np.nan)
        log_hl = np.log(high / safe_low)
        pv = (log_hl ** 2) / (4.0 * np.log(2.0))
        roll_pv = pv.rolling(window, min_periods=5).mean()
        return np.sqrt(roll_pv * 252.0).fillna(0.0)
