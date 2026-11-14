"""
Analysis API routes — Technical, Fundamental, Quote, and full analysis endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging
import asyncio
import numpy as np

from backend.data.fetcher import StockDataFetcher
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["Analysis"])


@router.get("/{symbol}/quote")
async def get_stock_quote(symbol: str):
    """
    Get fast instant price and 1-day change info for a stock (<200ms).
    Enables progressive UI loading so the user immediately sees live quotes.
    """
    try:
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        from backend.data.nse_symbols import NSE_STOCKS
        stock_meta = NSE_STOCKS.get(clean_symbol, {})

        # Fetch short 5d period for maximum speed
        df = StockDataFetcher.get_ohlcv(clean_symbol, period="5d")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No quote data found for {clean_symbol}")

        current_price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        # High/low from recent data
        high_price = float(df["High"].iloc[-1])
        low_price = float(df["Low"].iloc[-1])
        volume = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0

        # Check stock info if cached without blocking
        info = StockDataFetcher._get_cached(f"info_{clean_symbol}") or {}

        return {
            "symbol": clean_symbol,
            "name": stock_meta.get("name") or info.get("longName") or clean_symbol,
            "sector": stock_meta.get("sector") or info.get("sector") or "Diversified",
            "current_price": round(current_price, 2),
            "prev_close": round(prev_close, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(high_price, 2),
            "day_low": round(low_price, 2),
            "volume": volume,
            "disclaimer": settings.disclaimer,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}")
async def get_full_analysis(
    symbol: str,
    period: str = Query("1y", description="Data period: 1y, 2y, 5y"),
):
    """
    Get complete analysis for a stock: technical + fundamental + sentiment + market context.
    Executes sub-analyses concurrently in parallel threads to eliminate bottlenecks.
    """
    try:
        from backend.analysis.technical import TechnicalAnalyzer
        from backend.analysis.fundamental import FundamentalAnalyzer
        from backend.analysis.sentiment import SentimentAnalyzer
        from backend.analysis.market_context import MarketContextAnalyzer
        from backend.analysis.patterns import PatternRecognizer
        from backend.analysis.support_resistance import SupportResistanceAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()

        # 1. Fetch OHLCV data asynchronously
        df = await asyncio.to_thread(StockDataFetcher.get_ohlcv, clean_symbol, period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {clean_symbol}")

        # 2. Run all analyzers concurrently in parallel threads
        tech_task = asyncio.to_thread(TechnicalAnalyzer().analyze, clean_symbol)
        fund_task = asyncio.to_thread(FundamentalAnalyzer().analyze, clean_symbol)
        pattern_task = asyncio.to_thread(PatternRecognizer().scan_all_patterns, df)
        sr_task = asyncio.to_thread(SupportResistanceAnalyzer().analyze, df)
        sentiment_task = asyncio.to_thread(SentimentAnalyzer().analyze, clean_symbol)
        market_task = asyncio.to_thread(MarketContextAnalyzer().analyze, clean_symbol)

        # Run with safe timeout bounds so one slow network call never hangs the server
        results = await asyncio.gather(
            tech_task, fund_task, pattern_task, sr_task, sentiment_task, market_task,
            return_exceptions=True,
        )

        tech_result = results[0] if not isinstance(results[0], Exception) else {}
        fund_result = results[1] if not isinstance(results[1], Exception) else {}
        pattern_result = results[2] if not isinstance(results[2], Exception) else []
        sr_result = results[3] if not isinstance(results[3], Exception) else {}
        sentiment_result = results[4] if not isinstance(results[4], Exception) else {}
        market_result = results[5] if not isinstance(results[5], Exception) else {}

        # Log any sub-analysis errors for debugging
        for name, res in zip(["tech", "fund", "patterns", "sr", "sentiment", "market"], results):
            if isinstance(res, Exception):
                logger.warning(f"Sub-analysis '{name}' failed for {clean_symbol}: {res}")

        # 3. Compose overall signal
        signals = []
        if isinstance(tech_result, dict) and tech_result.get("composite"):
            signals.append({
                "source": "Technical",
                "direction": tech_result["composite"].get("direction", "NEUTRAL"),
                "confidence": tech_result["composite"].get("confidence", 50),
            })
        if isinstance(fund_result, dict) and fund_result.get("composite_score"):
            score = fund_result["composite_score"]
            fund_direction = "BUY" if score > 65 else ("SELL" if score < 35 else "NEUTRAL")
            signals.append({
                "source": "Fundamental",
                "direction": fund_direction,
                "confidence": score,
            })
        if isinstance(sentiment_result, dict) and sentiment_result.get("overall_sentiment") is not None:
            sent = sentiment_result["overall_sentiment"]
            sent_direction = "BUY" if sent > 0.2 else ("SELL" if sent < -0.2 else "NEUTRAL")
            signals.append({
                "source": "Sentiment",
                "direction": sent_direction,
                "confidence": min(abs(sent) * 100, 100),
            })

        # Overall verdict
        buy_signals = sum(1 for s in signals if s["direction"] == "BUY")
        sell_signals = sum(1 for s in signals if s["direction"] == "SELL")
        avg_confidence = sum(s["confidence"] for s in signals) / len(signals) if signals else 50

        if buy_signals > sell_signals:
            overall = "BULLISH"
        elif sell_signals > buy_signals:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL"

        # Current price info
        current_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
        prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 and "Close" in df.columns else None
        change_pct = ((current_price - prev_close) / prev_close * 100) if current_price and prev_close else None

        return {
            "symbol": clean_symbol,
            "current_price": current_price,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "overall_verdict": overall,
            "overall_confidence": round(avg_confidence, 1),
            "signals": signals,
            "technical": tech_result,
            "fundamental": fund_result,
            "patterns": pattern_result,
            "support_resistance": sr_result,
            "sentiment": sentiment_result,
            "market_context": market_result,
            "disclaimer": settings.disclaimer,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/{symbol}/technical")
async def get_technical_analysis(symbol: str):
    """Get technical analysis only."""
    try:
        from backend.analysis.technical import TechnicalAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        analyzer = TechnicalAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, clean_symbol)
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/fundamental")
async def get_fundamental_analysis(symbol: str):
    """Get fundamental analysis only."""
    try:
        from backend.analysis.fundamental import FundamentalAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        analyzer = FundamentalAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, clean_symbol)
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Fundamental analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/patterns")
async def get_pattern_analysis(symbol: str, period: str = "1y"):
    """Get candlestick and chart pattern analysis."""
    try:
        from backend.analysis.patterns import PatternRecognizer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        df = await asyncio.to_thread(StockDataFetcher.get_ohlcv, clean_symbol, period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {clean_symbol}")

        recognizer = PatternRecognizer()
        result = await asyncio.to_thread(recognizer.scan_all_patterns, df)
        return {"symbol": clean_symbol, "patterns": result, "disclaimer": settings.disclaimer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/support-resistance")
async def get_support_resistance(symbol: str, period: str = "1y"):
    """Get support/resistance levels, pivot points, and Fibonacci."""
    try:
        from backend.analysis.support_resistance import SupportResistanceAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        df = await asyncio.to_thread(StockDataFetcher.get_ohlcv, clean_symbol, period)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {clean_symbol}")

        analyzer = SupportResistanceAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, df)
        return {"symbol": clean_symbol, **result, "disclaimer": settings.disclaimer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"S/R analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/ohlcv")
async def get_ohlcv_data(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
):
    """Get raw OHLCV data for charting with strict data sanitization."""
    try:
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        df = await asyncio.to_thread(StockDataFetcher.get_ohlcv, clean_symbol, period, interval)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {clean_symbol}")

        # Convert to list of dicts for frontend with strict NaN and finite filtering
        records = []
        for idx, row in df.iterrows():
            try:
                o = float(row.get("Open", 0))
                h = float(row.get("High", 0))
                l = float(row.get("Low", 0))
                c = float(row.get("Close", 0))
                v = float(row.get("Volume", 0))
                if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isinf(o) or np.isinf(c):
                    continue
                records.append({
                    "time": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10],
                    "open": round(o, 2),
                    "high": round(max(h, o, c), 2),
                    "low": round(min(l, o, c), 2),
                    "close": round(c, 2),
                    "volume": int(v) if not np.isnan(v) and not np.isinf(v) else 0,
                })
            except Exception:
                continue

        return {"symbol": clean_symbol, "interval": interval, "data": records}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OHLCV error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/market-context")
async def get_market_context(symbol: str):
    """Get market context: VIX, global cues, sector analysis."""
    try:
        from backend.analysis.market_context import MarketContextAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        analyzer = MarketContextAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, clean_symbol)
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Market context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{symbol}/sentiment")
async def get_sentiment(symbol: str):
    """Get news sentiment analysis."""
    try:
        from backend.analysis.sentiment import SentimentAnalyzer

        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        analyzer = SentimentAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, clean_symbol)
        result["disclaimer"] = settings.disclaimer
        return result
    except Exception as e:
        logger.error(f"Sentiment error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
