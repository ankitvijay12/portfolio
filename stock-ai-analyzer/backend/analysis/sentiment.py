"""
Sentiment Analysis Module for Stock AI Analyzer.
Analyzes news sentiment, market buzz, social momentum, and corporate actions
for Indian stocks with time-decay weighting and VADER polarity scoring.
"""

import logging
import math
import re
import numpy as np
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

from backend.data.fetcher import StockDataFetcher
from backend.data.nse_symbols import NSE_STOCKS, get_sector, get_stocks_by_sector, get_sector_index
from backend.config import settings

logger = logging.getLogger(__name__)

# Specialized Financial & Indian Market Sentiment Lexicon
FINANCIAL_LEXICON: Dict[str, float] = {
    # Bullish / Positive Financial Terms
    "surge": 2.4,
    "surges": 2.4,
    "surging": 2.4,
    "jump": 2.0,
    "jumps": 2.0,
    "rally": 2.3,
    "rallies": 2.3,
    "rallying": 2.3,
    "bullish": 2.5,
    "breakout": 2.2,
    "outperform": 2.2,
    "outperformed": 2.2,
    "multibagger": 3.0,
    "record high": 2.8,
    "all-time high": 2.8,
    "profit jump": 2.5,
    "earnings beat": 2.6,
    "beat estimates": 2.5,
    "revenue up": 2.0,
    "dividend": 1.5,
    "bonus issue": 2.0,
    "order win": 2.4,
    "contract win": 2.4,
    "acquisition": 1.5,
    "expansion": 1.6,
    "upgrade": 2.2,
    "upgraded": 2.2,
    "target raised": 2.2,
    "strong buy": 2.8,
    "buy rating": 2.0,
    "fii buying": 2.2,
    "dii buying": 2.0,
    "turnaround": 2.1,
    "margin expansion": 2.2,

    # Bearish / Negative Financial Terms
    "plunge": -2.6,
    "plunges": -2.6,
    "plunging": -2.6,
    "slump": -2.4,
    "slumps": -2.4,
    "crash": -3.2,
    "crashes": -3.2,
    "crashing": -3.2,
    "bearish": -2.5,
    "breakdown": -2.2,
    "underperform": -2.2,
    "underperformed": -2.2,
    "record low": -2.8,
    "all-time low": -2.8,
    "profit decline": -2.5,
    "earnings miss": -2.6,
    "misses estimates": -2.5,
    "revenue drop": -2.2,
    "loss widening": -2.8,
    "downgrade": -2.4,
    "downgraded": -2.4,
    "target cut": -2.2,
    "sell rating": -2.2,
    "fii selling": -2.2,
    "fraud": -3.6,
    "scam": -3.8,
    "probe": -2.5,
    "raid": -2.8,
    "default": -3.5,
    "debt crisis": -3.2,
    "promoter pledge": -2.0,
    "pledged shares": -1.8,
    "margin contraction": -2.2,
    "regulatory action": -2.4,
    "penalty": -2.2,
    "fine imposed": -2.0,
    "sebi notice": -2.5,
    "lawsuit": -2.2,
    "cfo resigns": -2.2,
    "auditor resigns": -3.0,
}


class SentimentAnalyzer:
    """
    Performs comprehensive sentiment analysis on NSE stocks.
    Evaluates:
      1. News Sentiment (Multi-tier news fetching with time-decay weighted VADER scoring)
      2. Market Buzz & Velocity (7-day / 30-day article volume, sentiment trajectory)
      3. Corporate Actions Tracking (Dividends, Splits, Bonus, Rights, Earnings dates)
      4. Composite Sentiment Score (0-100) with confidence and direction
    """

    TIME_DECAY_LAMBDA = 0.15  # Exponential decay constant (~4.6-day half life)

    def __init__(self):
        """Initialize sentiment analyzer and VADER intensity analyzer if available."""
        self.vader = self._init_vader()

    def _init_vader(self) -> Any:
        """Initializes NLTK VADER sentiment analyzer with custom financial lexicon."""
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                try:
                    nltk.download("vader_lexicon", quiet=True)
                except Exception as dl_err:
                    logger.debug(f"Could not auto-download vader_lexicon: {dl_err}")

            sia = SentimentIntensityAnalyzer()
            # Update with domain-specific Indian market lexicon
            sia.lexicon.update(FINANCIAL_LEXICON)
            logger.info("Initialized NLTK VADER analyzer with financial lexicon enrichment.")
            return sia
        except Exception as e:
            logger.warning(f"NLTK VADER not available ({e}); using built-in financial rule-based analyzer.")
            return None

    _sentiment_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    CACHE_TTL: float = 600.0  # 10 minutes

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Main entry point for sentiment analysis of a given stock.

        Args:
            symbol: NSE symbol (e.g. 'RELIANCE', 'TCS', or 'RELIANCE.NS')

        Returns:
            Dictionary containing news sentiment, market buzz, corporate actions,
            aggregate score, direction, and SEBI disclaimer.
        """
        import time
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        now = time.time()
        if clean_symbol in self._sentiment_cache:
            cache_time, cached_res = self._sentiment_cache[clean_symbol]
            if now - cache_time < self.CACHE_TTL:
                logger.info(f"Returning cached sentiment for {clean_symbol}")
                return cached_res

        company_info = NSE_STOCKS.get(clean_symbol, {})
        company_name = company_info.get("name", clean_symbol)
        sector = get_sector(clean_symbol)

        logger.info(f"Starting sentiment analysis for {clean_symbol} ({company_name})")

        try:
            # 1. Fetch News from multiple sources (NewsAPI, yfinance, Google News RSS)
            articles = self._fetch_news(clean_symbol, company_name)

            # 2. Score Articles with Time-Decay Weighting
            scored_articles, sentiment_summary = self._score_articles(articles)

            # 3. Market Buzz & Sentiment Trajectory
            buzz_metrics = self._evaluate_market_buzz(scored_articles)

            # 4. Corporate Actions Detection
            corporate_actions = self._evaluate_corporate_actions(clean_symbol)

            # 5. Composite Sentiment Score & Direction
            composite_result = self._calculate_composite_score(
                sentiment_summary=sentiment_summary,
                buzz_metrics=buzz_metrics,
                corporate_actions=corporate_actions,
            )

            result = {
                "symbol": clean_symbol,
                "company_name": company_name,
                "sector": sector,
                "signal_type": "SENTIMENT",
                "direction": composite_result["direction"],
                "sentiment_score": composite_result["score"],
                "overall_sentiment": round(composite_result["score"] / 50.0 - 1.0, 2),  # normalized -1 to +1
                "confidence": composite_result["confidence"],
                "sentiment_label": composite_result["label"],
                "summary": composite_result["summary"],
                "news_sentiment": {
                    "total_articles": len(scored_articles),
                    "positive_count": sentiment_summary["positive_count"],
                    "neutral_count": sentiment_summary["neutral_count"],
                    "negative_count": sentiment_summary["negative_count"],
                    "positive_pct": sentiment_summary["positive_pct"],
                    "negative_pct": sentiment_summary["negative_pct"],
                    "neutral_pct": sentiment_summary["neutral_pct"],
                    "raw_compound_average": sentiment_summary["raw_compound_avg"],
                    "time_decay_compound_average": sentiment_summary["decay_compound_avg"],
                    "top_articles": scored_articles[:12],  # Provide top 12 recent articles
                },
                "market_buzz": buzz_metrics,
                "corporate_actions": corporate_actions,
                "sebi_disclaimer": settings.disclaimer,
            }
            SentimentAnalyzer._sentiment_cache[clean_symbol] = (now, result)
            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {clean_symbol}: {e}", exc_info=True)
            return self._generate_fallback_response(clean_symbol, company_name, str(e))

    # -------------------------------------------------------------------------
    # News Ingestion (Multi-tier Fetcher)
    # -------------------------------------------------------------------------
    def _fetch_news(self, symbol: str, company_name: str) -> List[Dict[str, Any]]:
        """
        Fetches news from available sources with fallback:
          Tier 1: NewsAPI if news_api_key is set
          Tier 2: yfinance ticker news
          Tier 3: Google News RSS for Indian Equities
        """
        articles: List[Dict[str, Any]] = []

        # Tier 1: NewsAPI (if configured)
        if settings.news_api_key:
            try:
                api_articles = self._fetch_newsapi(symbol, company_name)
                articles.extend(api_articles)
                logger.info(f"Fetched {len(api_articles)} articles from NewsAPI for {symbol}")
            except Exception as e:
                logger.debug(f"NewsAPI fetch failed: {e}")

        # Tier 2: yfinance direct news
        try:
            yf_articles = self._fetch_yfinance_news(symbol)
            articles.extend(yf_articles)
            logger.info(f"Fetched {len(yf_articles)} articles from yfinance for {symbol}")
        except Exception as e:
            logger.debug(f"yfinance news fetch failed: {e}")

        # Tier 3: Google News RSS for Indian Equities (ensures fresh India-specific coverage)
        if len(articles) < 10:
            try:
                rss_articles = self._fetch_google_news_rss(symbol, company_name)
                articles.extend(rss_articles)
                logger.info(f"Fetched {len(rss_articles)} articles from Google News RSS for {symbol}")
            except Exception as e:
                logger.debug(f"Google News RSS fetch failed: {e}")

        # Deduplicate and sort descending by publish date
        unique_articles = self._deduplicate_articles(articles)
        unique_articles.sort(key=lambda x: x["published_at"], reverse=True)

        return unique_articles

    def _fetch_newsapi(self, symbol: str, company_name: str) -> List[Dict[str, Any]]:
        """Fetches news from newsapi.org using HTTP request."""
        query = f'"{company_name}" OR "{symbol} share" OR "{symbol} stock"'
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 25,
            "apiKey": settings.news_api_key,
        }
        url = f"https://newsapi.org/v2/everything?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

        import json

        with urllib.request.urlopen(req, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        if data.get("status") == "ok":
            for item in data.get("articles", []):
                pub_str = item.get("publishedAt")
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")) if pub_str else datetime.now(timezone.utc)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

                results.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", "") or "",
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url", ""),
                    "published_at": pub_dt,
                })
        return results

    def _fetch_yfinance_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetches news items from yfinance Ticker."""
        ticker = StockDataFetcher.get_ticker(symbol)
        raw_news = getattr(ticker, "news", [])
        if not raw_news:
            return []

        results = []
        for item in raw_news:
            # Handle yfinance news schema
            title = item.get("title", "")
            publisher = item.get("publisher", "Financial News")
            link = item.get("link", "")
            pub_time = item.get("providerPublishTime")

            if pub_time:
                pub_dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)
            else:
                pub_dt = datetime.now(timezone.utc)

            # Sometimes yfinance wraps details in 'content' dict
            content = item.get("content", {})
            if isinstance(content, dict) and not title:
                title = content.get("title", "")
                link = content.get("canonicalUrl", {}).get("url", link)

            if title:
                results.append({
                    "title": title.strip(),
                    "description": "",
                    "source": publisher,
                    "url": link,
                    "published_at": pub_dt,
                })
        return results

    def _fetch_google_news_rss(self, symbol: str, company_name: str) -> List[Dict[str, Any]]:
        """Fetches India-specific news from Google News RSS feed."""
        query = f"{company_name} share price NSE India"
        encoded_query = urllib.parse.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        req = urllib.request.Request(
            rss_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )

        with urllib.request.urlopen(req, timeout=4) as response:
            xml_data = response.read()

        root = ET.fromstring(xml_data)
        results = []

        # Parse RSS items
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            link_elem = item.find("link")
            pub_date_elem = item.find("pubDate")
            source_elem = item.find("source")

            title = title_elem.text if title_elem is not None and title_elem.text else ""
            link = link_elem.text if link_elem is not None and link_elem.text else ""
            source = source_elem.text if source_elem is not None and source_elem.text else "Google News"

            pub_dt = datetime.now(timezone.utc)
            if pub_date_elem is not None and pub_date_elem.text:
                try:
                    # e.g., 'Sat, 07 Nov 2026 09:30:00 GMT'
                    from email.utils import parsedate_to_datetime
                    pub_dt = parsedate_to_datetime(pub_date_elem.text)
                except Exception:
                    pub_dt = datetime.now(timezone.utc)

            if title:
                # Clean Google News suffix (e.g. "Title - The Economic Times")
                clean_title = re.sub(r" - [^-]+$", "", title).strip()
                results.append({
                    "title": clean_title,
                    "description": "",
                    "source": source,
                    "url": link,
                    "published_at": pub_dt,
                })

        return results

    @staticmethod
    def _deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates articles based on normalized title or URL."""
        seen_titles = set()
        seen_urls = set()
        unique = []

        for art in articles:
            title_norm = re.sub(r"[^a-zA-Z0-9]", "", art["title"].lower())
            url = art.get("url", "").strip()

            if not title_norm or len(title_norm) < 10:
                continue

            if title_norm in seen_titles:
                continue
            if url and url in seen_urls:
                continue

            seen_titles.add(title_norm)
            if url:
                seen_urls.add(url)
            unique.append(art)

        return unique

    # -------------------------------------------------------------------------
    # Sentiment Scoring with Time Decay
    # -------------------------------------------------------------------------
    def _score_text(self, text: str) -> Dict[str, float]:
        """Scores a piece of text using VADER or built-in financial rule-based scorer."""
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

        if self.vader:
            try:
                return self.vader.polarity_scores(text)
            except Exception:
                pass

        # Built-in financial sentiment fallback
        return self._rule_based_sentiment(text)

    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Fallback financial sentiment evaluator using domain keywords."""
        words = re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())
        if not words:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

        score = 0.0
        pos_words = 0
        neg_words = 0

        # Check multi-word and single-word phrases
        lowered = text.lower()
        for phrase, weight in FINANCIAL_LEXICON.items():
            if " " in phrase and phrase in lowered:
                score += weight
                if weight > 0:
                    pos_words += 1
                else:
                    neg_words += 1

        for w in words:
            if w in FINANCIAL_LEXICON and " " not in w:
                weight = FINANCIAL_LEXICON[w]
                score += weight
                if weight > 0:
                    pos_words += 1
                else:
                    neg_words += 1

        # Normalize score into [-1.0, 1.0] using hyperbolic tangent approximation
        compound = float(np.tanh(score / 3.0)) if score != 0 else 0.0
        total_matched = pos_words + neg_words
        if total_matched > 0:
            pos_ratio = pos_words / (total_matched + len(words) * 0.1)
            neg_ratio = neg_words / (total_matched + len(words) * 0.1)
            neu_ratio = max(0.0, 1.0 - (pos_ratio + neg_ratio))
        else:
            pos_ratio, neg_ratio, neu_ratio = 0.0, 0.0, 1.0

        return {
            "compound": round(compound, 4),
            "pos": round(pos_ratio, 3),
            "neu": round(neu_ratio, 3),
            "neg": round(neg_ratio, 3),
        }

    def _score_articles(self, articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Calculates polarity scores for all articles and applies exponential time decay.
        Formula: weight = exp(-lambda * days_ago)
        """
        now = datetime.now(timezone.utc)
        scored: List[Dict[str, Any]] = []

        total_weight = 0.0
        weighted_compound_sum = 0.0
        raw_compounds: List[float] = []

        pos_count = 0
        neg_count = 0
        neu_count = 0

        for art in articles:
            pub_at = art["published_at"]
            if pub_at.tzinfo is None:
                pub_at = pub_at.replace(tzinfo=timezone.utc)

            # Calculate age in days
            age_seconds = max(0.0, (now - pub_at).total_seconds())
            days_ago = age_seconds / 86400.0

            # Exponential decay weight
            weight = math.exp(-self.TIME_DECAY_LAMBDA * days_ago)

            # Combined text for polarity evaluation (title + description)
            combined_text = f"{art['title']}. {art.get('description', '')}".strip()
            polarity = self._score_text(combined_text)
            compound = polarity["compound"]

            # Tag classification
            if compound >= 0.08:
                sentiment_tag = "POSITIVE"
                pos_count += 1
            elif compound <= -0.08:
                sentiment_tag = "NEGATIVE"
                neg_count += 1
            else:
                sentiment_tag = "NEUTRAL"
                neu_count += 1

            raw_compounds.append(compound)
            total_weight += weight
            weighted_compound_sum += compound * weight

            scored.append({
                "title": art["title"],
                "source": art.get("source", "Unknown"),
                "url": art.get("url", ""),
                "published_at": pub_at.isoformat(),
                "days_ago": round(days_ago, 1),
                "decay_weight": round(weight, 3),
                "compound": round(compound, 3),
                "pos": polarity["pos"],
                "neu": polarity["neu"],
                "neg": polarity["neg"],
                "sentiment_tag": sentiment_tag,
            })

        total_articles = len(articles)
        if total_articles > 0:
            raw_avg = float(np.mean(raw_compounds))
            decay_avg = weighted_compound_sum / total_weight if total_weight > 0 else raw_avg
            pos_pct = round((pos_count / total_articles) * 100.0, 1)
            neg_pct = round((neg_count / total_articles) * 100.0, 1)
            neu_pct = round((neu_count / total_articles) * 100.0, 1)
        else:
            raw_avg, decay_avg = 0.0, 0.0
            pos_pct, neg_pct, neu_pct = 0.0, 0.0, 100.0

        summary = {
            "total_articles": total_articles,
            "positive_count": pos_count,
            "neutral_count": neu_count,
            "negative_count": neg_count,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "raw_compound_avg": round(raw_avg, 3),
            "decay_compound_avg": round(decay_avg, 3),
        }

        return scored, summary

    # -------------------------------------------------------------------------
    # Market Buzz & Sentiment Trajectory
    # -------------------------------------------------------------------------
    def _evaluate_market_buzz(self, scored_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measures volume of media coverage over 7d and 30d, and determines
        whether media sentiment is improving or worsening.
        """
        articles_7d = [a for a in scored_articles if a["days_ago"] <= 7.0]
        articles_30d = [a for a in scored_articles if a["days_ago"] <= 30.0]
        older_articles = [a for a in scored_articles if 7.0 < a["days_ago"] <= 30.0]

        count_7d = len(articles_7d)
        count_30d = len(articles_30d)

        # Buzz Level & Score
        if count_30d >= 15 or count_7d >= 6:
            buzz_level = "HIGH"
            buzz_score = 90.0
            buzz_desc = f"Elevated news and media coverage ({count_7d} in 7d, {count_30d} in 30d); stock is in high focus."
        elif count_30d >= 5 or count_7d >= 2:
            buzz_level = "MODERATE"
            buzz_score = 65.0
            buzz_desc = f"Moderate media presence ({count_7d} in 7d, {count_30d} in 30d); regular market interest."
        else:
            buzz_level = "LOW"
            buzz_score = 40.0
            buzz_desc = f"Low media velocity ({count_7d} in 7d, {count_30d} in 30d); stock is flying under the radar."

        # Sentiment Trend Calculation
        if count_7d >= 2 and len(older_articles) >= 2:
            recent_avg = float(np.mean([a["compound"] for a in articles_7d]))
            older_avg = float(np.mean([a["compound"] for a in older_articles]))
            diff = recent_avg - older_avg

            if diff >= 0.12:
                trend = "IMPROVING"
                trend_desc = f"Recent news sentiment is improving (+{diff:.2f} delta vs previous weeks)."
            elif diff <= -0.12:
                trend = "DETERIORATING"
                trend_desc = f"Recent news sentiment is turning cautious ({diff:.2f} delta vs previous weeks)."
            else:
                trend = "STABLE"
                trend_desc = "Sentiment remains consistent with recent baseline."
        elif len(scored_articles) >= 4:
            # Split articles chronologically into halves
            half = len(scored_articles) // 2
            first_half_avg = float(np.mean([a["compound"] for a in scored_articles[:half]]))
            second_half_avg = float(np.mean([a["compound"] for a in scored_articles[half:]]))
            diff = first_half_avg - second_half_avg
            if diff >= 0.10:
                trend = "IMPROVING"
                trend_desc = "Sentiment trend has an upward trajectory."
            elif diff <= -0.10:
                trend = "DETERIORATING"
                trend_desc = "Sentiment trend has a downward trajectory."
            else:
                trend = "STABLE"
                trend_desc = "Sentiment trajectory is stable."
        else:
            trend = "STABLE"
            trend_desc = "Insufficient multi-week data for trend differential."

        return {
            "articles_last_7_days": count_7d,
            "articles_last_30_days": count_30d,
            "buzz_level": buzz_level,
            "buzz_score": buzz_score,
            "sentiment_trend": trend,
            "trend_description": trend_desc,
            "interpretation": buzz_desc,
        }

    # -------------------------------------------------------------------------
    # Corporate Actions Tracking
    # -------------------------------------------------------------------------
    def _evaluate_corporate_actions(self, symbol: str) -> Dict[str, Any]:
        """
        Extracts corporate actions (dividends, splits, bonuses, earnings calendar)
        from yfinance and categorizes their market impact.
        """
        ticker = StockDataFetcher.get_ticker(symbol)
        actions_list: List[Dict[str, Any]] = []

        try:
            # 1. Check Calendar (Earnings Dates, Dividend Dates)
            calendar = getattr(ticker, "calendar", None)
            if calendar is not None:
                # Calendar might be dict or DataFrame
                cal_dict = calendar if isinstance(calendar, dict) else (calendar.to_dict() if hasattr(calendar, "to_dict") else {})
                # Look for earnings date
                earnings_date = cal_dict.get("Earnings Date") or cal_dict.get("Earnings High")
                if earnings_date:
                    ed_str = str(earnings_date[0] if isinstance(earnings_date, list) and earnings_date else earnings_date)
                    actions_list.append({
                        "action_type": "EARNINGS_RELEASE",
                        "date": ed_str[:10],
                        "impact": "NEUTRAL",
                        "description": f"Quarterly earnings announcement scheduled for {ed_str[:10]}; potential volatility catalyst.",
                    })

                div_date = cal_dict.get("Dividend Date") or cal_dict.get("Ex-Dividend Date")
                if div_date:
                    dd_str = str(div_date[0] if isinstance(div_date, list) and div_date else div_date)
                    actions_list.append({
                        "action_type": "DIVIDEND_EVENT",
                        "date": dd_str[:10],
                        "impact": "POSITIVE",
                        "description": f"Dividend ex-date on {dd_str[:10]}; rewarding shareholders.",
                    })
        except Exception as e:
            logger.debug(f"Calendar extraction error: {e}")

        try:
            # 2. Check Recent Dividends
            divs = getattr(ticker, "dividends", None)
            if divs is not None and not divs.empty:
                latest_div_date = str(divs.index[-1])[:10]
                latest_div_val = float(divs.iloc[-1])
                actions_list.append({
                    "action_type": "CASH_DIVIDEND",
                    "date": latest_div_date,
                    "impact": "POSITIVE",
                    "description": f"Dividend of ₹{latest_div_val:.2f} per share declared on {latest_div_date}.",
                })
        except Exception as e:
            logger.debug(f"Dividends extraction error: {e}")

        try:
            # 3. Check Stock Splits
            splits = getattr(ticker, "splits", None)
            if splits is not None and not splits.empty:
                latest_split_date = str(splits.index[-1])[:10]
                split_ratio = float(splits.iloc[-1])
                if split_ratio > 1.0:
                    actions_list.append({
                        "action_type": "STOCK_SPLIT",
                        "date": latest_split_date,
                        "impact": "POSITIVE",
                        "description": f"Stock split executed ({split_ratio}:1) on {latest_split_date}; enhances retail trading liquidity.",
                    })
        except Exception as e:
            logger.debug(f"Splits extraction error: {e}")

        # Summary of net corporate action posture
        pos_actions = sum(1 for a in actions_list if a["impact"] == "POSITIVE")
        neg_actions = sum(1 for a in actions_list if a["impact"] == "NEGATIVE")

        if pos_actions > neg_actions:
            overall_impact = "POSITIVE"
            act_summary = "Corporate actions demonstrate shareholder-friendly capital allocation (dividends/splits)."
        elif neg_actions > pos_actions:
            overall_impact = "NEGATIVE"
            act_summary = "Corporate developments suggest capital dilution or restructuring headwinds."
        else:
            overall_impact = "NEUTRAL"
            act_summary = "Corporate actions and board announcements remain standard."

        return {
            "overall_impact": overall_impact,
            "actions_count": len(actions_list),
            "summary": act_summary,
            "actions": actions_list[:5],  # Top 5 actions
        }

    # -------------------------------------------------------------------------
    # Composite Sentiment Score & Direction
    # -------------------------------------------------------------------------
    def _calculate_composite_score(
        self,
        sentiment_summary: Dict[str, Any],
        buzz_metrics: Dict[str, Any],
        corporate_actions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesizes news compound score, sentiment trend, buzz volume, and corporate actions
        into an overall sentiment score (0-100) and trading direction ("BUY", "NEUTRAL", "SELL").
        """
        decay_compound = sentiment_summary.get("decay_compound_avg", 0.0)
        total_articles = sentiment_summary.get("total_articles", 0)

        # Base score normalized from compound [-1.0, 1.0] to [0.0, 100.0]
        # A compound of 0.0 maps to 50.0 (Neutral). +0.5 maps to 75.0. -0.5 maps to 25.0.
        base_score = ((decay_compound + 1.0) / 2.0) * 100.0

        # Adjust for Sentiment Trend (+4 for IMPROVING, -4 for DETERIORATING)
        trend_adj = 4.0 if buzz_metrics.get("sentiment_trend") == "IMPROVING" else (-4.0 if buzz_metrics.get("sentiment_trend") == "DETERIORATING" else 0.0)

        # Adjust for Corporate Actions (+3 for POSITIVE, -3 for NEGATIVE)
        action_adj = 3.0 if corporate_actions.get("overall_impact") == "POSITIVE" else (-3.0 if corporate_actions.get("overall_impact") == "NEGATIVE" else 0.0)

        final_score = max(0.0, min(100.0, base_score + trend_adj + action_adj))
        final_score = round(final_score, 1)

        # Confidence calculation
        # Confidence increases with article sample size and consistency
        if total_articles >= 15:
            conf_articles = 85.0
        elif total_articles >= 8:
            conf_articles = 70.0
        elif total_articles >= 3:
            conf_articles = 55.0
        elif total_articles >= 1:
            conf_articles = 40.0
        else:
            conf_articles = 20.0

        # High buzz boosts confidence slightly; zero coverage reduces confidence
        confidence = conf_articles
        if buzz_metrics.get("buzz_level") == "HIGH":
            confidence = min(95.0, confidence + 5.0)

        # Direction mapping
        if final_score >= 62.0:
            direction = "BUY"
            label = "BULLISH"
            summary = "Constructive media sentiment with positive news flow and improving market narrative."
        elif final_score <= 38.0:
            direction = "SELL"
            label = "BEARISH"
            summary = "Prevailing negative news coverage or corporate overhangs creating sentiment headwinds."
        else:
            direction = "NEUTRAL"
            label = "NEUTRAL"
            summary = "Balanced media sentiment with no overwhelming directional catalyst."

        return {
            "score": final_score,
            "confidence": round(confidence, 1),
            "direction": direction,
            "label": label,
            "summary": summary,
        }

    def _generate_fallback_response(self, symbol: str, company_name: str, reason: str) -> Dict[str, Any]:
        """Provides a graceful fallback when live sentiment analysis encounters network or parse errors."""
        return {
            "symbol": symbol,
            "company_name": company_name,
            "sector": get_sector(symbol),
            "signal_type": "SENTIMENT",
            "direction": "NEUTRAL",
            "sentiment_score": 50.0,
            "confidence": 20.0,
            "sentiment_label": "NEUTRAL",
            "summary": f"Sentiment analysis limited: {reason}",
            "news_sentiment": {
                "total_articles": 0,
                "positive_count": 0,
                "neutral_count": 0,
                "negative_count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 100.0,
                "raw_compound_average": 0.0,
                "time_decay_compound_average": 0.0,
                "top_articles": [],
            },
            "market_buzz": {
                "articles_last_7_days": 0,
                "articles_last_30_days": 0,
                "buzz_level": "LOW",
                "buzz_score": 40.0,
                "sentiment_trend": "STABLE",
                "trend_description": "Baseline media presence.",
                "interpretation": "Limited news coverage detected.",
            },
            "corporate_actions": {
                "overall_impact": "NEUTRAL",
                "actions_count": 0,
                "summary": "No upcoming corporate action alerts.",
                "actions": [],
            },
            "sebi_disclaimer": settings.disclaimer,
        }
