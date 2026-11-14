"""
Stock data fetcher using yfinance.
Handles NSE/BSE symbol normalization, OHLCV data, fundamentals, and option chains.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches stock data from yfinance with NSE symbol normalization and caching."""

    _cache: Dict[str, Any] = {}
    _cache_expiry: Dict[str, datetime] = {}
    CACHE_TTL_MINUTES = 15

    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """Convert plain symbol or company name to yfinance NSE format."""
        if not symbol:
            return "RELIANCE.NS"
        symbol = symbol.strip()
        if symbol.startswith("^") or symbol.endswith("=F") or symbol.endswith("=X"):
            return symbol

        upper_sym = symbol.upper()
        if upper_sym.endswith(".NS") or upper_sym.endswith(".BO"):
            return upper_sym

        # Check known dictionary first
        from backend.data.nse_symbols import NSE_STOCKS
        if upper_sym in NSE_STOCKS:
            return f"{upper_sym}.NS"

        # Check if symbol matches name in NSE_STOCKS
        for sym, info in NSE_STOCKS.items():
            if upper_sym in info.get("name", "").upper():
                return f"{sym}.NS"

        # If query has spaces or special characters, search via yfinance
        if " " in symbol or len(symbol) > 10:
            try:
                search = yf.Search(symbol)
                for q in search.quotes:
                    sym = q.get("symbol", "")
                    if sym.endswith(".NS") or sym.endswith(".BO") or q.get("exchange") in ("NSI", "BSE"):
                        return sym
            except Exception as e:
                logger.warning(f"Symbol search error for '{symbol}': {e}")

        # Remove special characters from ticker
        clean = re.sub(r'[^A-Za-z0-9\-\.\^]', '', upper_sym)
        return f"{clean}.NS"

    @classmethod
    def _get_cached(cls, key: str) -> Optional[Any]:
        """Return cached data if still valid."""
        if key in cls._cache and key in cls._cache_expiry:
            if datetime.now() < cls._cache_expiry[key]:
                return cls._cache[key]
            del cls._cache[key]
            del cls._cache_expiry[key]
        return None

    @classmethod
    def _set_cache(cls, key: str, data: Any):
        """Cache data with TTL."""
        cls._cache[key] = data
        cls._cache_expiry[key] = datetime.now() + timedelta(minutes=cls.CACHE_TTL_MINUTES)

    @classmethod
    def get_ticker(cls, symbol: str) -> yf.Ticker:
        """Get yfinance Ticker object."""
        yf_symbol = cls.normalize_symbol(symbol)
        return yf.Ticker(yf_symbol)

    @classmethod
    def get_ohlcv(
        cls,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a stock.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            period: Data period (e.g., "1y", "2y", "5y", "10y", "max")
            interval: Data interval (e.g., "1d", "1wk", "1mo", "15m", "1h")
            start: Start date (YYYY-MM-DD), overrides period
            end: End date (YYYY-MM-DD), overrides period

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        cache_key = f"ohlcv_{symbol}_{period}_{interval}_{start}_{end}"
        cached = cls._get_cached(cache_key)
        if cached is not None:
            return cached

        yf_symbol = cls.normalize_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)

        try:
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df is None or df.empty:
                logger.warning(f"No data returned for {yf_symbol}")
                return pd.DataFrame()

            # Handle multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names
            df.columns = [str(c).title().replace(" ", "_") for c in df.columns]
            if "Stock_Splits" in df.columns:
                df = df.drop(columns=["Stock_Splits", "Dividends"], errors="ignore")

            # Remove timezone from DatetimeIndex
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Drop duplicate timestamps and sort ascending
            df = df[~df.index.duplicated(keep="last")].sort_index()

            # Drop any rows with NaN in essential OHLC columns
            expected = ["Open", "High", "Low", "Close"]
            for col in expected:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} for {yf_symbol}")
                    return pd.DataFrame()

            df = df.dropna(subset=expected)

            # Ensure Volume column exists and is finite
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
            else:
                df["Volume"] = 0

            # Enforce High >= max(Open, Close) and Low <= min(Open, Close)
            df["High"] = df[["High", "Open", "Close"]].max(axis=1)
            df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

            cls._set_cache(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {yf_symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def get_multiple_timeframes(cls, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multi-timeframe analysis.

        Returns:
            Dict with keys: 'weekly', 'daily', '15min' (15min only for recent 60 days)
        """
        return {
            "weekly": cls.get_ohlcv(symbol, period="5y", interval="1wk"),
            "daily": cls.get_ohlcv(symbol, period="2y", interval="1d"),
            "15min": cls.get_ohlcv(symbol, period="60d", interval="15m"),
        }

    @classmethod
    def get_stock_info(cls, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive stock info (fundamentals, sector, etc.).

        Returns:
            Dict with all available stock info fields.
        """
        cache_key = f"info_{symbol}"
        cached = cls._get_cached(cache_key)
        if cached is not None:
            return cached

        ticker = cls.get_ticker(symbol)
        try:
            info = ticker.info
            cls._set_cache(cache_key, info)
            return info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}

    @classmethod
    def get_fundamentals(cls, symbol: str) -> Dict[str, Any]:
        """
        Extract key fundamental metrics from stock info.

        Returns:
            Dict with P/E, P/B, ROE, ROCE, debt ratios, margins, growth, etc.
        """
        info = cls.get_stock_info(symbol)
        if not info:
            return {}

        def safe_get(key: str, default=None):
            return info.get(key, default)

        fundamentals = {
            # Valuation
            "pe_ratio": safe_get("trailingPE"),
            "forward_pe": safe_get("forwardPE"),
            "pb_ratio": safe_get("priceToBook"),
            "peg_ratio": safe_get("pegRatio"),
            "ev_ebitda": safe_get("enterpriseToEbitda"),
            "market_cap": safe_get("marketCap"),
            "enterprise_value": safe_get("enterpriseValue"),

            # Profitability
            "roe": safe_get("returnOnEquity"),
            "roa": safe_get("returnOnAssets"),
            "profit_margin": safe_get("profitMargins"),
            "operating_margin": safe_get("operatingMargins"),
            "gross_margin": safe_get("grossMargins"),

            # Growth
            "revenue_growth": safe_get("revenueGrowth"),
            "earnings_growth": safe_get("earningsGrowth"),
            "eps_trailing": safe_get("trailingEps"),
            "eps_forward": safe_get("forwardEps"),

            # Financial Health
            "debt_to_equity": safe_get("debtToEquity"),
            "current_ratio": safe_get("currentRatio"),
            "quick_ratio": safe_get("quickRatio"),
            "total_debt": safe_get("totalDebt"),
            "total_cash": safe_get("totalCash"),

            # Cash Flow
            "free_cash_flow": safe_get("freeCashflow"),
            "operating_cash_flow": safe_get("operatingCashflow"),

            # Dividends
            "dividend_yield": safe_get("dividendYield"),
            "dividend_rate": safe_get("dividendRate"),
            "payout_ratio": safe_get("payoutRatio"),

            # Shareholding
            "held_by_institutions": safe_get("heldPercentInstitutions"),
            "held_by_insiders": safe_get("heldPercentInsiders"),

            # Basic info
            "sector": safe_get("sector"),
            "industry": safe_get("industry"),
            "full_name": safe_get("longName"),
            "currency": safe_get("currency", "INR"),
            "exchange": safe_get("exchange"),
        }

        return {k: v for k, v in fundamentals.items() if v is not None}

    @classmethod
    def get_financial_statements(cls, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch income statement, balance sheet, and cash flow statement.

        Returns:
            Dict with keys: 'income_stmt', 'balance_sheet', 'cash_flow' (quarterly and annual)
        """
        ticker = cls.get_ticker(symbol)
        try:
            return {
                "income_stmt": ticker.income_stmt,
                "income_stmt_quarterly": ticker.quarterly_income_stmt,
                "balance_sheet": ticker.balance_sheet,
                "balance_sheet_quarterly": ticker.quarterly_balance_sheet,
                "cash_flow": ticker.cashflow,
                "cash_flow_quarterly": ticker.quarterly_cashflow,
            }
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {}

    @classmethod
    def get_holders(cls, symbol: str) -> Dict[str, Any]:
        """Fetch institutional and major holders data."""
        ticker = cls.get_ticker(symbol)
        try:
            result = {}
            try:
                result["major_holders"] = ticker.major_holders
            except Exception:
                result["major_holders"] = pd.DataFrame()
            try:
                result["institutional_holders"] = ticker.institutional_holders
            except Exception:
                result["institutional_holders"] = pd.DataFrame()
            return result
        except Exception as e:
            logger.error(f"Error fetching holders for {symbol}: {e}")
            return {}

    @classmethod
    def get_option_chain(cls, symbol: str) -> Dict[str, Any]:
        """
        Fetch option chain data for PCR, max pain, OI analysis.

        Returns:
            Dict with 'calls', 'puts' DataFrames, 'expirations' list, and computed metrics.
        """
        ticker = cls.get_ticker(symbol)
        try:
            expirations = ticker.options
            if not expirations:
                return {"available": False}

            # Get nearest expiry
            nearest = expirations[0]
            chain = ticker.option_chain(nearest)

            calls = chain.calls
            puts = chain.puts

            # Compute Put-Call Ratio
            total_call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            total_put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            # Compute Max Pain
            max_pain = cls._compute_max_pain(calls, puts)

            return {
                "available": True,
                "expiration": nearest,
                "all_expirations": list(expirations),
                "calls": calls.to_dict("records") if not calls.empty else [],
                "puts": puts.to_dict("records") if not puts.empty else [],
                "pcr": round(pcr, 2),
                "max_pain": max_pain,
                "total_call_oi": int(total_call_oi),
                "total_put_oi": int(total_put_oi),
            }
        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {"available": False, "error": str(e)}

    @staticmethod
    def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
        """
        Compute max pain strike price — the price at which total option losses are maximized
        for option buyers (or equivalently, minimized for option writers).
        """
        try:
            strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
            min_pain = float("inf")
            max_pain_strike = None

            for strike in strikes:
                # Pain for call holders: sum of (strike_price - test_price) * OI for ITM calls
                call_pain = 0
                for _, row in calls.iterrows():
                    if strike > row["strike"]:
                        call_pain += (strike - row["strike"]) * row.get("openInterest", 0)

                # Pain for put holders
                put_pain = 0
                for _, row in puts.iterrows():
                    if strike < row["strike"]:
                        put_pain += (row["strike"] - strike) * row.get("openInterest", 0)

                total_pain = call_pain + put_pain
                if total_pain < min_pain:
                    min_pain = total_pain
                    max_pain_strike = strike

            return max_pain_strike
        except Exception:
            return None

    @classmethod
    def get_nifty50_data(cls, period: str = "1y") -> pd.DataFrame:
        """Fetch Nifty 50 index data for relative strength calculation."""
        return cls.get_ohlcv("^NSEI", period=period)

    @classmethod
    def get_india_vix(cls, period: str = "1y") -> pd.DataFrame:
        """Fetch India VIX data."""
        return cls.get_ohlcv("^INDIAVIX", period=period)

    @classmethod
    def get_global_cues(cls) -> Dict[str, pd.DataFrame]:
        """Fetch global market data for context analysis."""
        symbols = {
            "sp500": "^GSPC",
            "nasdaq": "^IXIC",
            "crude_oil": "CL=F",
            "usd_inr": "INR=X",
            "us_10y_yield": "^TNX",
            "gold": "GC=F",
        }
        result = {}
        for name, sym in symbols.items():
            try:
                df = yf.Ticker(sym).history(period="6mo")
                if not df.empty:
                    result[name] = df
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({sym}): {e}")
        return result

    @classmethod
    def search_stocks(cls, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol."""
        from backend.data.nse_symbols import NSE_STOCKS

        query = query.upper()
        results = []
        for sym, info in NSE_STOCKS.items():
            if query in sym or query in info.get("name", "").upper():
                results.append({"symbol": sym, **info})
                if len(results) >= limit:
                    break
        return results
