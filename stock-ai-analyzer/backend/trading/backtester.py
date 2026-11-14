"""
Strategy Backtesting Engine for Stock AI Analyzer.

Performs vectorized and event-driven backtesting for technical, rule-based,
and machine-learning strategies on Indian equity markets.
Applies realistic transaction friction: flat brokerage (₹20/order),
Securities Transaction Tax (STT 0.1% on sell side), and slippage (0.05%).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# pandas-ta indicator library
try:
    import pandas_ta as ta
except ImportError:
    ta = None

from backend.data.fetcher import StockDataFetcher
from backend.config import settings

logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Simulates trading strategies on historical NSE/BSE stock data.
    Computes institutional performance metrics including Sharpe ratio,
    win rate, profit factor, max drawdown, and equity curves.
    """

    STRATEGY_MAP = {
        "ema_crossover": "EMA Crossover",
        "EMA Crossover": "EMA Crossover",
        "rsi_reversal": "RSI Reversal",
        "RSI Reversal": "RSI Reversal",
        "supertrend": "Supertrend",
        "Supertrend": "Supertrend",
        "breakout": "Breakout",
        "20-Day Breakout": "Breakout",
        "Breakout": "Breakout",
        "macd": "MACD Crossover",
        "macd_crossover": "MACD Crossover",
        "MACD Crossover": "MACD Crossover",
        "ml_signal": "ML Signal",
        "ML Signal": "ML Signal",
    }

    def __init__(
        self,
        brokerage: Optional[float] = None,
        stt_rate: Optional[float] = None,
        slippage_pct: Optional[float] = None,
        risk_free_rate: float = 0.06,  # 6% annual risk-free rate for India
    ):
        """
        Initialize backtesting engine with realistic Indian market friction.

        Args:
            brokerage: Flat brokerage fee per executed order in INR (default: ₹20)
            stt_rate: Securities Transaction Tax rate on sell value (default: 0.1% = 0.001)
            slippage_pct: Execution slippage percentage (default: 0.05% = 0.0005)
            risk_free_rate: Annualized risk-free rate for Sharpe computation (default: 6%)
        """
        self.brokerage = brokerage if brokerage is not None else getattr(settings, "brokerage_per_trade", 20.0)
        self.stt_rate = stt_rate if stt_rate is not None else getattr(settings, "stt_rate", 0.001)
        # slippage_pct from settings is typically 0.05 for 0.05% -> decimal 0.0005
        raw_slip = slippage_pct if slippage_pct is not None else getattr(settings, "slippage_pct", 0.05)
        self.slippage = raw_slip / 100.0 if raw_slip >= 0.01 else raw_slip
        self.risk_free_rate = risk_free_rate
        self.fetcher = StockDataFetcher

    def backtest(
        self,
        symbol: str,
        strategy: str = "EMA Crossover",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        capital: float = 1000000.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy on a stock ticker.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            strategy: Strategy name or ID
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            capital: Initial portfolio capital in INR (default: 10 Lakh INR)
            kwargs: Strategy specific parameters (fast_ema, slow_ema, etc.)

        Returns:
            Dictionary containing performance metrics, trade log, equity curve,
            and confidence score.
        """
        clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "").strip()
        strat_key = self.STRATEGY_MAP.get(strategy, strategy)

        # 1. Fetch OHLCV data
        period = "5y" if not (start_date and end_date) else None
        df = self.fetcher.get_ohlcv(
            clean_symbol,
            period=period or "5y",
            interval="1d",
            start=start_date,
            end=end_date,
        )

        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient historical data for backtesting {clean_symbol}")
            return self._empty_result(clean_symbol, strat_key, capital, start_date, end_date)

        # Ensure index is sorted datetime
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # 2. Generate Strategy Signals
        df = self._generate_signals(df, strat_key, **kwargs)

        # 3. Simulate Trades and Portfolio Equity
        trades_list, equity_curve = self._simulate_trading(df, capital)

        # 4. Convert trades to DataFrame and apply costs
        trades_df = pd.DataFrame(trades_list)
        if not trades_df.empty:
            trades_df = self._apply_costs(trades_df)

        # 5. Compute Quantitative Metrics
        metrics = self._compute_metrics(trades_df, equity_curve, capital)

        # Format period string
        p_start = df.index[0].strftime("%Y-%m-%d")
        p_end = df.index[-1].strftime("%Y-%m-%d")
        period_str = f"{p_start} to {p_end}"

        # Trade list for frontend display
        formatted_trades = []
        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                formatted_trades.append({
                    "entry_date": row["entry_date"].strftime("%Y-%m-%d") if hasattr(row["entry_date"], "strftime") else str(row["entry_date"]),
                    "entry_price": round(float(row["entry_price"]), 2),
                    "exit_date": row["exit_date"].strftime("%Y-%m-%d") if hasattr(row["exit_date"], "strftime") else str(row["exit_date"]),
                    "exit_price": round(float(row["exit_price"]), 2),
                    "quantity": int(row["quantity"]),
                    "pnl": round(float(row["net_pnl"]), 2),
                    "return_pct": round(float(row["return_pct"]), 2),
                    "holding_days": int(row["holding_days"]),
                    "exit_reason": str(row.get("exit_reason", "Signal Exit")),
                })

        return {
            "strategy": strat_key,
            "symbol": clean_symbol,
            "period": period_str,
            "total_trades": metrics["total_trades"],
            "profitable_trades": metrics["profitable_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "avg_win": metrics["avg_win"],
            "avg_loss": metrics["avg_loss"],
            "total_return_pct": metrics["total_return_pct"],
            "final_equity": metrics["final_equity"],
            "confidence": metrics["confidence"],
            "equity_curve": equity_curve,
            "trades": formatted_trades,
            "disclaimer": settings.disclaimer,
            "sebi_disclaimer": settings.disclaimer,
        }

    def backtest_all_strategies(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        capital: float = 1000000.0,
    ) -> List[Dict[str, Any]]:
        """
        Run all standard strategies across the historical data for comparison.

        Strategies executed:
        1. EMA Crossover
        2. RSI Reversal
        3. Supertrend
        4. Breakout
        5. MACD Crossover
        6. ML Signal

        Args:
            symbol: Stock symbol
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            capital: Initial capital

        Returns:
            List of backtest result dictionaries sorted by Sharpe ratio descending.
        """
        unique_strategies = [
            "EMA Crossover",
            "RSI Reversal",
            "Supertrend",
            "Breakout",
            "MACD Crossover",
            "ML Signal",
        ]

        results = []
        for strat in unique_strategies:
            try:
                res = self.backtest(
                    symbol=symbol,
                    strategy=strat,
                    start_date=start_date,
                    end_date=end_date,
                    capital=capital,
                )
                results.append(res)
            except Exception as e:
                logger.error(f"Strategy {strat} failed on {symbol}: {e}")

        # Sort by Sharpe ratio descending
        results.sort(key=lambda x: x.get("sharpe_ratio", 0.0), reverse=True)
        return results

    def get_equity_curve(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract equity curve for frontend charting.

        Args:
            result: Return dictionary from backtest()

        Returns:
            List of {"date": "YYYY-MM-DD", "equity": float} objects.
        """
        return result.get("equity_curve", [])

    def _apply_costs(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Indian transaction friction:
        1. Flat brokerage ₹20 on buy + ₹20 on sell
        2. STT (Securities Transaction Tax) of 0.1% on sell side turnover
        3. Slippage 0.05% already factored into effective entry/exit execution.

        Args:
            trades_df: DataFrame with raw trades

        Returns:
            DataFrame with net_pnl, total_costs, and net return_pct.
        """
        df = trades_df.copy()

        # Buy turnover = quantity * entry_price
        buy_turnover = df["quantity"] * df["entry_price"]
        # Sell turnover = quantity * exit_price
        sell_turnover = df["quantity"] * df["exit_price"]

        # Brokerage = ₹20 buy + ₹20 sell
        brokerage_total = self.brokerage * 2.0
        # STT = 0.1% on sell turnover
        stt_total = sell_turnover * self.stt_rate

        total_costs = brokerage_total + stt_total
        gross_pnl = sell_turnover - buy_turnover
        net_pnl = gross_pnl - total_costs

        df["gross_pnl"] = gross_pnl
        df["total_costs"] = total_costs
        df["net_pnl"] = net_pnl
        # Return on capital deployed for the trade
        df["return_pct"] = np.where(buy_turnover > 0, (net_pnl / buy_turnover) * 100.0, 0.0)

        return df

    def _compute_metrics(
        self,
        trades_df: pd.DataFrame,
        equity_curve: List[Dict[str, Any]],
        capital: float,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive quantitative trading metrics.

        Args:
            trades_df: Completed trades DataFrame with costs
            equity_curve: Portfolio value history
            capital: Initial investment capital

        Returns:
            Metrics dictionary with win rate, profit factor, drawdown, Sharpe, etc.
        """
        total_trades = len(trades_df)

        if total_trades == 0 or trades_df.empty:
            return {
                "total_trades": 0,
                "profitable_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "total_return_pct": 0.0,
                "final_equity": capital,
                "confidence": 0.0,
            }

        winning_trades = trades_df[trades_df["net_pnl"] > 0]
        losing_trades = trades_df[trades_df["net_pnl"] < 0]

        profitable_count = len(winning_trades)
        win_rate = round((profitable_count / total_trades) * 100.0, 1)

        gross_profit = float(winning_trades["net_pnl"].sum()) if not winning_trades.empty else 0.0
        gross_loss = float(abs(losing_trades["net_pnl"].sum())) if not losing_trades.empty else 0.0

        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        else:
            profit_factor = 99.9 if gross_profit > 0 else 0.0

        avg_win = round(float(winning_trades["return_pct"].mean()), 2) if not winning_trades.empty else 0.0
        avg_loss = round(float(losing_trades["return_pct"].mean()), 2) if not losing_trades.empty else 0.0

        # Equity Curve Analysis
        eq_values = [p["equity"] for p in equity_curve]
        final_equity = eq_values[-1] if eq_values else capital
        total_return_pct = round(((final_equity - capital) / capital) * 100.0, 1)

        # Max Drawdown calculation
        eq_series = pd.Series(eq_values)
        peak = eq_series.cummax()
        drawdowns = (eq_series - peak) / peak * 100.0
        max_drawdown = round(float(drawdowns.min()), 1) if not drawdowns.empty else 0.0

        # Annualized Sharpe Ratio calculation (252 Indian trading days)
        daily_returns = eq_series.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            daily_rf = self.risk_free_rate / 252.0
            excess_daily = daily_returns - daily_rf
            sharpe = (excess_daily.mean() / daily_returns.std()) * np.sqrt(252.0)
            sharpe_ratio = round(float(sharpe), 2)
        else:
            sharpe_ratio = 0.0

        # Confidence Score (0-100%) based on sample size and stability
        sample_conf = min(50.0, total_trades * 1.5)  # 30+ trades gives 45-50%
        sharpe_conf = max(0.0, min(30.0, sharpe_ratio * 15.0))  # Sharpe 2.0 gives 30%
        win_conf = max(0.0, min(20.0, (win_rate - 30.0) * 0.5))  # Win rate > 50 gives 10-20%
        confidence = round(min(100.0, sample_conf + sharpe_conf + win_conf), 1)

        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_return_pct": total_return_pct,
            "final_equity": round(final_equity, 2),
            "confidence": confidence,
        }

    def _generate_signals(self, df: pd.DataFrame, strategy: str, **kwargs: Any) -> pd.DataFrame:
        """
        Generate vectorized entry (+1) and exit (-1) signals based on selected strategy.
        """
        df = df.copy()
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        df["signal"] = 0  # 1 = Buy, -1 = Sell, 0 = Hold

        if strategy == "EMA Crossover":
            fast_p = kwargs.get("fast_ema", 20)
            slow_p = kwargs.get("slow_ema", 50)
            ema_fast = close.ewm(span=fast_p, adjust=False).mean()
            ema_slow = close.ewm(span=slow_p, adjust=False).mean()

            # Buy when fast crosses above slow; Sell when fast crosses below slow
            bullish_cross = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
            bearish_cross = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
            df.loc[bullish_cross, "signal"] = 1
            df.loc[bearish_cross, "signal"] = -1

        elif strategy == "RSI Reversal":
            rsi_period = kwargs.get("rsi_period", 14)
            oversold = kwargs.get("rsi_oversold", 30)
            overbought = kwargs.get("rsi_overbought", 70)

            rsi = self._compute_rsi(close, rsi_period)
            # Buy when RSI crosses above 30 from below (oversold reversal)
            buy_sig = (rsi > oversold) & (rsi.shift(1) <= oversold)
            # Sell when RSI crosses below 70 from above (overbought exhaustion)
            sell_sig = (rsi < overbought) & (rsi.shift(1) >= overbought)
            df.loc[buy_sig, "signal"] = 1
            df.loc[sell_sig, "signal"] = -1

        elif strategy == "Supertrend":
            period = kwargs.get("period", 10)
            multiplier = kwargs.get("multiplier", 3.0)
            direction = self._compute_supertrend(high, low, close, period, multiplier)
            # Buy when direction flips to 1; Sell when flips to -1
            buy_sig = (direction == 1) & (direction.shift(1) == -1)
            sell_sig = (direction == -1) & (direction.shift(1) == 1)
            df.loc[buy_sig, "signal"] = 1
            df.loc[sell_sig, "signal"] = -1

        elif strategy == "Breakout":
            lookback = kwargs.get("lookback", 20)
            vol_mult = kwargs.get("volume_threshold", 1.5)

            prior_high = high.shift(1).rolling(lookback).max()
            prior_low = low.shift(1).rolling(lookback).min()
            vol_avg = volume.shift(1).rolling(lookback).mean()

            # Breakout condition: Price breaks 20-day high with volume > 1.5x avg
            breakout_buy = (close > prior_high) & (volume > (vol_mult * vol_avg))
            # Exit when price closes below 10-day low or 20 EMA
            exit_sell = (close < prior_low)
            df.loc[breakout_buy, "signal"] = 1
            df.loc[exit_sell, "signal"] = -1

        elif strategy == "MACD Crossover":
            fast_p = kwargs.get("fast", 12)
            slow_p = kwargs.get("slow", 26)
            signal_p = kwargs.get("signal", 9)

            ema_fast = close.ewm(span=fast_p, adjust=False).mean()
            ema_slow = close.ewm(span=slow_p, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
            hist = macd_line - signal_line

            buy_sig = (hist > 0) & (hist.shift(1) <= 0)
            sell_sig = (hist < 0) & (hist.shift(1) >= 0)
            df.loc[buy_sig, "signal"] = 1
            df.loc[sell_sig, "signal"] = -1

        elif strategy == "ML Signal":
            # Probability-based quantitative signal
            rsi = self._compute_rsi(close, 14)
            ret_5d = close.pct_change(5)
            ema_20 = close.ewm(span=20, adjust=False).mean()
            trend_dist = (close - ema_20) / ema_20

            # Logistic probability function simulating ML up-probability
            z = ((rsi - 50.0) / 10.0) + (ret_5d * 15.0) + (trend_dist * 10.0)
            prob_up = 1.0 / (1.0 + np.exp(-np.clip(z, -6.0, 6.0)))

            buy_sig = (prob_up > 0.60) & (prob_up.shift(1) <= 0.60)
            sell_sig = (prob_up < 0.40) & (prob_up.shift(1) >= 0.40)
            df.loc[buy_sig, "signal"] = 1
            df.loc[sell_sig, "signal"] = -1

        return df

    def _simulate_trading(
        self,
        df: pd.DataFrame,
        initial_capital: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Simulate trade execution bar-by-bar with slippage and portfolio tracking.
        """
        cash = initial_capital
        position_shares = 0
        entry_price_eff = 0.0
        entry_date = None

        trades = []
        equity_curve = []

        close_prices = df["Close"].values
        signals = df["signal"].values
        dates = df.index

        for i in range(len(df)):
            current_date = dates[i]
            bar_close = close_prices[i]
            sig = signals[i]

            # 1. Buy Execution
            if sig == 1 and position_shares == 0:
                # Add slippage to entry price
                effective_entry = round(bar_close * (1.0 + self.slippage), 2)
                # Available cash after reserving entry brokerage
                available_for_shares = cash - self.brokerage
                if available_for_shares > effective_entry:
                    position_shares = int(available_for_shares // effective_entry)
                    if position_shares > 0:
                        total_outlay = (position_shares * effective_entry) + self.brokerage
                        cash -= total_outlay
                        entry_price_eff = effective_entry
                        entry_date = current_date

            # 2. Sell Execution
            elif sig == -1 and position_shares > 0:
                # Deduct slippage from exit price
                effective_exit = round(bar_close * (1.0 - self.slippage), 2)
                gross_proceeds = position_shares * effective_exit
                stt = round(gross_proceeds * self.stt_rate, 2)
                net_proceeds = gross_proceeds - stt - self.brokerage
                cash += net_proceeds

                holding_days = (current_date - entry_date).days if entry_date else 1

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price_eff,
                    "exit_date": current_date,
                    "exit_price": effective_exit,
                    "quantity": position_shares,
                    "holding_days": max(1, holding_days),
                    "exit_reason": "Strategy Signal Exit",
                })

                position_shares = 0
                entry_price_eff = 0.0
                entry_date = None

            # 3. Daily Portfolio Mark-to-Market Valuation
            current_position_value = (position_shares * bar_close) if position_shares > 0 else 0.0
            daily_equity = cash + current_position_value

            equity_curve.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "equity": round(daily_equity, 2),
            })

        # Settle any remaining open position at final bar for clean accounting
        if position_shares > 0:
            final_date = dates[-1]
            effective_exit = round(close_prices[-1] * (1.0 - self.slippage), 2)
            gross_proceeds = position_shares * effective_exit
            stt = round(gross_proceeds * self.stt_rate, 2)
            net_proceeds = gross_proceeds - stt - self.brokerage
            cash += net_proceeds
            holding_days = (final_date - entry_date).days if entry_date else 1

            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price_eff,
                "exit_date": final_date,
                "exit_price": effective_exit,
                "quantity": position_shares,
                "holding_days": max(1, holding_days),
                "exit_reason": "End of Backtest Period",
            })
            if equity_curve:
                equity_curve[-1]["equity"] = round(cash, 2)

        return trades, equity_curve

    def _compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized Relative Strength Index computation."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _compute_supertrend(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.Series:
        """Vectorized ATR-based Supertrend calculation."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().bfill()

        hl2 = (high + low) / 2.0
        upper_basic = hl2 + (multiplier * atr)
        lower_basic = hl2 - (multiplier * atr)

        final_upper = upper_basic.copy()
        final_lower = lower_basic.copy()
        direction = pd.Series(1, index=close.index)

        c_arr = close.values
        ub_arr = upper_basic.values
        lb_arr = lower_basic.values
        fu_arr = final_upper.values
        fl_arr = final_lower.values
        dir_arr = np.ones(len(close), dtype=int)

        for i in range(1, len(close)):
            # Final upper band
            if c_arr[i - 1] <= fu_arr[i - 1]:
                fu_arr[i] = min(ub_arr[i], fu_arr[i - 1])
            else:
                fu_arr[i] = ub_arr[i]

            # Final lower band
            if c_arr[i - 1] >= fl_arr[i - 1]:
                fl_arr[i] = max(lb_arr[i], fl_arr[i - 1])
            else:
                fl_arr[i] = lb_arr[i]

            # Direction
            if c_arr[i] > fu_arr[i - 1]:
                dir_arr[i] = 1
            elif c_arr[i] < fl_arr[i - 1]:
                dir_arr[i] = -1
            else:
                dir_arr[i] = dir_arr[i - 1]

        return pd.Series(dir_arr, index=close.index)

    def _empty_result(
        self,
        symbol: str,
        strategy: str,
        capital: float,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Dict[str, Any]:
        """Return standardized zero result when data is insufficient."""
        return {
            "strategy": strategy,
            "symbol": symbol,
            "period": f"{start_date or 'N/A'} to {end_date or 'N/A'}",
            "total_trades": 0,
            "profitable_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_return_pct": 0.0,
            "final_equity": capital,
            "confidence": 0.0,
            "equity_curve": [],
            "trades": [],
            "disclaimer": settings.disclaimer,
            "sebi_disclaimer": settings.disclaimer,
        }
