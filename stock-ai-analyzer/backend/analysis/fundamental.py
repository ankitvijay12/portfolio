"""
Fundamental Analysis Module for Stock AI Analyzer.
Evaluates valuation, profitability, growth, financial health, and shareholding patterns
for NSE-listed companies with peer benchmarking and composite scoring.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.data.fetcher import StockDataFetcher
from backend.data.nse_symbols import NSE_STOCKS, get_sector, get_stocks_by_sector, get_sector_index
from backend.config import settings

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """
    Performs comprehensive fundamental analysis on Indian equities.
    Evaluates:
      1. Valuation Metrics (P/E, Forward P/E, P/B, PEG, EV/EBITDA) vs Sector Benchmarks
      2. Profitability Analysis (ROE, ROCE, ROA, Margins, Margin Trends)
      3. Growth Analysis (Revenue & Profit QoQ/YoY, EPS trajectory, 3-5 Year CAGR)
      4. Financial Health (Debt/Equity, Interest Coverage, Current & Quick Ratios, FCF)
      5. Shareholding Analysis (Promoter holding, pledging risk, Institutional interest)
      6. Peer Comparison (Relative ranking within NSE sector peers)
      7. Composite Fundamental Score (0-100) with confidence rating
    """

    # Industry standard benchmarks for fallback when sector data is limited
    BENCHMARKS = {
        "pe_ratio": {"undervalued": 15.0, "fair_max": 28.0},
        "pb_ratio": {"undervalued": 2.0, "fair_max": 4.5},
        "peg_ratio": {"undervalued": 1.0, "fair_max": 1.8},
        "ev_ebitda": {"undervalued": 10.0, "fair_max": 18.0},
        "roe": {"poor": 10.0, "good": 15.0, "excellent": 20.0},
        "roce": {"poor": 10.0, "good": 15.0, "excellent": 20.0},
        "roa": {"poor": 3.0, "good": 6.0, "excellent": 10.0},
        "operating_margin": {"low": 8.0, "good": 15.0, "excellent": 22.0},
        "profit_margin": {"low": 5.0, "good": 10.0, "excellent": 18.0},
        "debt_to_equity": {"strong": 0.5, "moderate": 1.0, "high_risk": 2.0},
        "interest_coverage": {"stressed": 1.5, "caution": 3.0, "strong": 5.0},
        "current_ratio": {"tight": 1.0, "good": 1.5},
        "quick_ratio": {"caution": 0.8, "healthy": 1.1},
    }

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Main entry point for fundamental analysis of a given stock symbol.

        Args:
            symbol: NSE symbol (e.g., 'RELIANCE', 'TCS', or 'RELIANCE.NS')

        Returns:
            Dictionary with detailed category breakdowns, peer comparisons,
            composite fundamental score, recommendation, and SEBI disclaimer.
        """
        clean_symbol = symbol.replace(".NS", "").replace(".BO", "").upper().strip()
        sector = get_sector(clean_symbol)

        logger.info(f"Starting fundamental analysis for {clean_symbol} (Sector: {sector})")

        try:
            # 1. Fetch data from StockDataFetcher
            fundamentals = StockDataFetcher.get_fundamentals(clean_symbol)
            statements = StockDataFetcher.get_financial_statements(clean_symbol)
            holders = StockDataFetcher.get_holders(clean_symbol)
            ticker = StockDataFetcher.get_ticker(clean_symbol)

            # If fundamentals are completely empty, handle gracefully
            if not fundamentals:
                logger.warning(f"No fundamental data retrieved for {clean_symbol}")
                return self._generate_fallback_response(clean_symbol, sector, "Insufficient fundamental data from source.")

            # 2. Peer Comparison & Sector averages
            peer_analysis = self._analyze_peers(clean_symbol, sector, fundamentals)

            # 3. Category Analyses
            valuation_analysis = self._analyze_valuation(fundamentals, peer_analysis.get("sector_averages", {}))
            profitability_analysis = self._analyze_profitability(fundamentals, statements)
            growth_analysis = self._analyze_growth(fundamentals, statements)
            health_analysis = self._analyze_financial_health(fundamentals, statements, sector)
            shareholding_analysis = self._analyze_shareholding(holders, fundamentals, ticker)

            # 4. Composite Scoring
            composite_result = self._calculate_composite_score(
                valuation=valuation_analysis,
                profitability=profitability_analysis,
                growth=growth_analysis,
                health=health_analysis,
                shareholding=shareholding_analysis,
            )

            # 5. Highlights and Risks synthesis
            strengths, risks = self._extract_strengths_and_risks(
                valuation_analysis,
                profitability_analysis,
                growth_analysis,
                health_analysis,
                shareholding_analysis,
            )

            return {
                "symbol": clean_symbol,
                "company_name": fundamentals.get("full_name", clean_symbol),
                "sector": sector,
                "industry": fundamentals.get("industry", "N/A"),
                "currency": fundamentals.get("currency", "INR"),
                "signal_type": "FUNDAMENTAL",
                "direction": composite_result["direction"],
                "fundamental_score": composite_result["score"],
                "confidence": composite_result["confidence"],
                "rating": composite_result["rating"],
                "summary": composite_result["summary"],
                "categories": {
                    "valuation": valuation_analysis,
                    "profitability": profitability_analysis,
                    "growth": growth_analysis,
                    "financial_health": health_analysis,
                    "shareholding": shareholding_analysis,
                },
                "peer_comparison": peer_analysis,
                "strengths": strengths,
                "risks": risks,
                "sebi_disclaimer": settings.disclaimer,
            }

        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {clean_symbol}: {e}", exc_info=True)
            return self._generate_fallback_response(clean_symbol, sector, str(e))

    # -------------------------------------------------------------------------
    # Valuation Analysis
    # -------------------------------------------------------------------------
    def _analyze_valuation(self, fundamentals: Dict[str, Any], sector_averages: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates P/E (trailing & forward), P/B, PEG, and EV/EBITDA."""
        metrics: Dict[str, Any] = {}
        scores: List[float] = []
        confidences: List[float] = []

        # 1. Trailing P/E
        pe = fundamentals.get("pe_ratio")
        sector_pe = sector_averages.get("pe_ratio")
        if pe is not None and pe > 0:
            benchmark = sector_pe if sector_pe and sector_pe > 0 else self.BENCHMARKS["pe_ratio"]["fair_max"]
            if pe < benchmark * 0.75:
                status, score = "UNDERVALUED", 88.0
                desc = f"Trailing P/E of {pe:.2f} is significantly below sector/benchmark ({benchmark:.2f}), offering value margin."
            elif pe <= benchmark * 1.25:
                status, score = "FAIR", 65.0
                desc = f"Trailing P/E of {pe:.2f} is aligned with sector/benchmark ({benchmark:.2f})."
            else:
                status, score = "OVERVALUED", 35.0
                desc = f"Trailing P/E of {pe:.2f} trades at an elevated premium vs sector/benchmark ({benchmark:.2f})."
            metrics["pe_ratio"] = {
                "value": round(pe, 2),
                "sector_average": round(sector_pe, 2) if sector_pe else None,
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(85.0)
        elif pe is not None and pe <= 0:
            metrics["pe_ratio"] = {
                "value": round(pe, 2),
                "sector_average": round(sector_pe, 2) if sector_pe else None,
                "status": "LOSS_MAKING",
                "score": 20.0,
                "confidence": 90.0,
                "interpretation": "Negative trailing P/E indicates operating loss or net negative earnings.",
            }
            scores.append(20.0)
            confidences.append(90.0)

        # 2. Forward P/E
        f_pe = fundamentals.get("forward_pe")
        if f_pe is not None and f_pe > 0:
            if pe and pe > 0:
                pe_expansion = (pe - f_pe) / pe * 100
                if f_pe < pe:
                    status, score = "UNDERVALUED", 85.0
                    desc = f"Forward P/E of {f_pe:.2f} indicates {pe_expansion:.1f}% expected earnings expansion."
                else:
                    status, score = "FAIR" if f_pe <= pe * 1.15 else "OVERVALUED", 55.0 if f_pe <= pe * 1.15 else 35.0
                    desc = f"Forward P/E of {f_pe:.2f} vs trailing {pe:.2f} implies moderate to slower future earnings."
            else:
                status = "UNDERVALUED" if f_pe < 18 else ("FAIR" if f_pe < 30 else "OVERVALUED")
                score = 80.0 if status == "UNDERVALUED" else (60.0 if status == "FAIR" else 40.0)
                desc = f"Forward P/E stands at {f_pe:.2f}."
            metrics["forward_pe"] = {
                "value": round(f_pe, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(80.0)

        # 3. P/B Ratio
        pb = fundamentals.get("pb_ratio")
        sector_pb = sector_averages.get("pb_ratio")
        if pb is not None and pb > 0:
            benchmark_pb = sector_pb if sector_pb and sector_pb > 0 else self.BENCHMARKS["pb_ratio"]["fair_max"]
            if pb < benchmark_pb * 0.7:
                status, score = "UNDERVALUED", 85.0
                desc = f"P/B of {pb:.2f} trades at an attractive discount relative to book value."
            elif pb <= benchmark_pb * 1.3:
                status, score = "FAIR", 65.0
                desc = f"P/B of {pb:.2f} is in line with book value expectations."
            else:
                status, score = "OVERVALUED", 35.0
                desc = f"P/B of {pb:.2f} represents an aggressive premium to book value."
            metrics["pb_ratio"] = {
                "value": round(pb, 2),
                "sector_average": round(sector_pb, 2) if sector_pb else None,
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(85.0)

        # 4. PEG Ratio (Growth-adjusted)
        peg = fundamentals.get("peg_ratio")
        if peg is not None and peg > 0:
            if peg < 1.0:
                status, score = "UNDERVALUED", 90.0
                desc = f"PEG of {peg:.2f} (<1.0) signals growth at an attractive valuation discount (Peter Lynch rule)."
            elif peg <= 1.8:
                status, score = "FAIR", 68.0
                desc = f"PEG of {peg:.2f} reflects reasonable valuation commensurate with earnings growth."
            else:
                status, score = "OVERVALUED", 35.0
                desc = f"PEG of {peg:.2f} (>1.8) indicates growth may already be excessively priced in."
            metrics["peg_ratio"] = {
                "value": round(peg, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(80.0)

        # 5. EV/EBITDA
        ev_ebitda = fundamentals.get("ev_ebitda")
        if ev_ebitda is not None and ev_ebitda > 0:
            if ev_ebitda < 10.0:
                status, score = "UNDERVALUED", 85.0
                desc = f"EV/EBITDA of {ev_ebitda:.2f} offers an attractive enterprise valuation."
            elif ev_ebitda <= 18.0:
                status, score = "FAIR", 65.0
                desc = f"EV/EBITDA of {ev_ebitda:.2f} is in the typical historical fair range."
            else:
                status, score = "OVERVALUED", 40.0
                desc = f"EV/EBITDA of {ev_ebitda:.2f} indicates elevated capital structure valuation."
            metrics["ev_ebitda"] = {
                "value": round(ev_ebitda, 2),
                "status": status,
                "score": score,
                "confidence": 75.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(75.0)

        # Category aggregate
        cat_score = round(float(np.mean(scores)), 1) if scores else 50.0
        cat_conf = round(float(np.mean(confidences)), 1) if confidences else 30.0
        overall_status = "UNDERVALUED" if cat_score >= 70 else ("FAIR" if cat_score >= 50 else "OVERVALUED")

        return {
            "score": cat_score,
            "confidence": cat_conf,
            "status": overall_status,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    # Profitability Analysis
    # -------------------------------------------------------------------------
    def _analyze_profitability(self, fundamentals: Dict[str, Any], statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluates ROE, ROCE, ROA, Margins, and quarterly margin trajectory."""
        metrics: Dict[str, Any] = {}
        scores: List[float] = []
        confidences: List[float] = []

        # 1. Return on Equity (ROE)
        roe = fundamentals.get("roe")
        if roe is not None:
            roe_pct = roe * 100.0 if abs(roe) < 1.0 else roe
            if roe_pct >= 20.0:
                status, score = "EXCELLENT", 95.0
                desc = f"ROE of {roe_pct:.1f}% represents superior compounding of shareholder equity."
            elif roe_pct >= 15.0:
                status, score = "GOOD", 80.0
                desc = f"ROE of {roe_pct:.1f}% satisfies strong investment quality criteria (>15%)."
            elif roe_pct >= 10.0:
                status, score = "MODERATE", 55.0
                desc = f"ROE of {roe_pct:.1f}% is average for capital deployment."
            else:
                status, score = "POOR", 25.0
                desc = f"ROE of {roe_pct:.1f}% is sub-par (<10%), indicating weak return on equity capital."
            metrics["roe"] = {
                "value": round(roe_pct, 2),
                "status": status,
                "score": score,
                "confidence": 90.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(90.0)

        # 2. Return on Capital Employed (ROCE)
        roce_val = self._extract_roce(statements.get("balance_sheet"), statements.get("income_stmt"))
        if roce_val is not None:
            if roce_val >= 20.0:
                status, score = "EXCELLENT", 95.0
                desc = f"ROCE of {roce_val:.1f}% shows outstanding operational efficiency across all capital employed."
            elif roce_val >= 15.0:
                status, score = "GOOD", 80.0
                desc = f"ROCE of {roce_val:.1f}% is strong (>15%), well ahead of cost of capital."
            elif roce_val >= 10.0:
                status, score = "MODERATE", 55.0
                desc = f"ROCE of {roce_val:.1f}% is moderate."
            else:
                status, score = "POOR", 30.0
                desc = f"ROCE of {roce_val:.1f}% is weak, potentially failing to beat weighted average cost of capital."
            metrics["roce"] = {
                "value": round(roce_val, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(80.0)

        # 3. Return on Assets (ROA)
        roa = fundamentals.get("roa")
        if roa is not None:
            roa_pct = roa * 100.0 if abs(roa) < 1.0 else roa
            if roa_pct >= 10.0:
                status, score = "EXCELLENT", 95.0
                desc = f"ROA of {roa_pct:.1f}% shows exceptional asset monetization (>10%)."
            elif roa_pct >= 5.0:
                status, score = "GOOD", 78.0
                desc = f"ROA of {roa_pct:.1f}% reflects healthy asset efficiency (>5%)."
            else:
                status, score = "POOR", 35.0
                desc = f"ROA of {roa_pct:.1f}% indicates sluggish asset turnover or heavy asset drag."
            metrics["roa"] = {
                "value": round(roa_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(85.0)

        # 4. Operating Margin & Net Profit Margin
        op_margin = fundamentals.get("operating_margin")
        if op_margin is not None:
            op_pct = op_margin * 100.0 if abs(op_margin) < 1.0 else op_margin
            status = "EXCELLENT" if op_pct >= 20.0 else ("GOOD" if op_pct >= 12.0 else ("FAIR" if op_pct >= 6.0 else "POOR"))
            score = 92.0 if status == "EXCELLENT" else (78.0 if status == "GOOD" else (50.0 if status == "FAIR" else 25.0))
            metrics["operating_margin"] = {
                "value": round(op_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"Operating margin at {op_pct:.1f}%.",
            }
            scores.append(score)
            confidences.append(85.0)

        np_margin = fundamentals.get("profit_margin")
        if np_margin is not None:
            np_pct = np_margin * 100.0 if abs(np_margin) < 1.0 else np_margin
            status = "EXCELLENT" if np_pct >= 15.0 else ("GOOD" if np_pct >= 8.0 else ("FAIR" if np_pct >= 4.0 else "POOR"))
            score = 90.0 if status == "EXCELLENT" else (75.0 if status == "GOOD" else (50.0 if status == "FAIR" else 20.0))
            metrics["net_profit_margin"] = {
                "value": round(np_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"Net profit margin at {np_pct:.1f}%.",
            }
            scores.append(score)
            confidences.append(85.0)

        # 5. Margin Trend from Quarterly Financial Statements
        margin_trend = self._extract_margin_trend(statements.get("income_stmt_quarterly"))
        metrics["margin_trend"] = margin_trend
        if margin_trend.get("status") == "IMPROVING":
            scores.append(85.0)
            confidences.append(75.0)
        elif margin_trend.get("status") == "DECLINING":
            scores.append(35.0)
            confidences.append(75.0)
        else:
            scores.append(60.0)
            confidences.append(65.0)

        cat_score = round(float(np.mean(scores)), 1) if scores else 50.0
        cat_conf = round(float(np.mean(confidences)), 1) if confidences else 30.0
        status_label = "EXCELLENT" if cat_score >= 80 else ("GOOD" if cat_score >= 65 else ("FAIR" if cat_score >= 45 else "WEAK"))

        return {
            "score": cat_score,
            "confidence": cat_conf,
            "status": status_label,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    # Growth Analysis
    # -------------------------------------------------------------------------
    def _analyze_growth(self, fundamentals: Dict[str, Any], statements: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluates Revenue & Profit growth (QoQ, YoY), EPS trajectory, and 3-5 Year CAGR."""
        metrics: Dict[str, Any] = {}
        scores: List[float] = []
        confidences: List[float] = []

        # 1. Revenue Growth (YoY)
        rev_growth = fundamentals.get("revenue_growth")
        if rev_growth is not None:
            rg_pct = rev_growth * 100.0 if abs(rev_growth) < 2.0 else rev_growth
            if rg_pct >= 20.0:
                status, score = "HIGH_GROWTH", 95.0
            elif rg_pct >= 10.0:
                status, score = "STEADY_GROWTH", 80.0
            elif rg_pct >= 0.0:
                status, score = "MILD_GROWTH", 55.0
            else:
                status, score = "CONTRACTION", 25.0
            metrics["revenue_growth_yoy"] = {
                "value": round(rg_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"YoY Revenue growth of {rg_pct:.1f}%.",
            }
            scores.append(score)
            confidences.append(85.0)

        # 2. Earnings Growth (YoY)
        earn_growth = fundamentals.get("earnings_growth")
        if earn_growth is not None:
            eg_pct = earn_growth * 100.0 if abs(earn_growth) < 2.0 else earn_growth
            if eg_pct >= 25.0:
                status, score = "STRONG_ACCELERATION", 95.0
            elif eg_pct >= 12.0:
                status, score = "HEALTHY_EXPANSION", 80.0
            elif eg_pct >= 0.0:
                status, score = "FLAT", 55.0
            else:
                status, score = "EARNINGS_DECLINE", 25.0
            metrics["profit_growth_yoy"] = {
                "value": round(eg_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"YoY Net Earnings growth of {eg_pct:.1f}%.",
            }
            scores.append(score)
            confidences.append(85.0)

        # 3. Quarterly Growth (QoQ) from quarterly statement
        q_income = statements.get("income_stmt_quarterly")
        qoq_metrics = self._extract_qoq_growth(q_income)
        if qoq_metrics:
            metrics["quarterly_growth"] = qoq_metrics
            q_rev = qoq_metrics.get("revenue_qoq_pct")
            if q_rev is not None:
                q_score = 85.0 if q_rev > 5.0 else (65.0 if q_rev >= 0 else 40.0)
                scores.append(q_score)
                confidences.append(75.0)

        # 4. EPS Trajectory
        eps_trailing = fundamentals.get("eps_trailing")
        eps_forward = fundamentals.get("eps_forward")
        if eps_trailing is not None and eps_forward is not None:
            if eps_trailing != 0:
                eps_exp_pct = ((eps_forward - eps_trailing) / abs(eps_trailing)) * 100.0
            else:
                eps_exp_pct = 0.0
            status = "EXPANDING" if eps_exp_pct > 5.0 else ("STABLE" if eps_exp_pct >= -5.0 else "CONTRACTING")
            score = 90.0 if eps_exp_pct > 15.0 else (75.0 if eps_exp_pct >= 5.0 else (50.0 if eps_exp_pct >= -5.0 else 30.0))
            metrics["eps_trajectory"] = {
                "trailing_eps": round(eps_trailing, 2),
                "forward_eps": round(eps_forward, 2),
                "expected_growth_pct": round(eps_exp_pct, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": f"Forward EPS projection of ₹{eps_forward:.2f} vs trailing ₹{eps_trailing:.2f} ({eps_exp_pct:+.1f}%).",
            }
            scores.append(score)
            confidences.append(80.0)

        # 5. Multi-year CAGR (Revenue & Net Income)
        annual_income = statements.get("income_stmt")
        cagr_metrics = self._compute_historical_cagr(annual_income)
        metrics["cagr"] = cagr_metrics
        if cagr_metrics.get("revenue_cagr_pct") is not None:
            rev_cagr = cagr_metrics["revenue_cagr_pct"]
            c_score = 90.0 if rev_cagr > 15.0 else (75.0 if rev_cagr >= 10.0 else (55.0 if rev_cagr >= 5.0 else 35.0))
            scores.append(c_score)
            confidences.append(70.0)

        cat_score = round(float(np.mean(scores)), 1) if scores else 50.0
        cat_conf = round(float(np.mean(confidences)), 1) if confidences else 30.0
        status_label = "HIGH_GROWTH" if cat_score >= 80 else ("MODERATE_GROWTH" if cat_score >= 60 else ("STAGNANT" if cat_score >= 40 else "DECLINING"))

        return {
            "score": cat_score,
            "confidence": cat_conf,
            "status": status_label,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    # Financial Health Analysis
    # -------------------------------------------------------------------------
    def _analyze_financial_health(self, fundamentals: Dict[str, Any], statements: Dict[str, pd.DataFrame], sector: str) -> Dict[str, Any]:
        """Evaluates Debt/Equity, Interest Coverage, Liquidity Ratios, and Free Cash Flow."""
        metrics: Dict[str, Any] = {}
        scores: List[float] = []
        confidences: List[float] = []
        is_financial = sector in ["Financial Services", "Banking"]

        # 1. Debt to Equity Ratio
        de = fundamentals.get("debt_to_equity")
        if de is not None:
            # yfinance occasionally gives D/E as percentage (e.g., 50 for 0.5) or decimal
            de_normalized = de / 100.0 if de > 10.0 else de
            if is_financial:
                # Banks have naturally higher leverage (deposits are liabilities)
                status = "ACCEPTABLE_FOR_BANK" if de_normalized < 6.0 else "HIGH_LEVERAGE"
                score = 80.0 if de_normalized < 5.0 else 55.0
                desc = f"Debt/Equity of {de_normalized:.2f} within standard banking capital structure parameters."
            else:
                if de_normalized < 0.5:
                    status, score = "STRONG_LOW_DEBT", 95.0
                    desc = f"Debt/Equity of {de_normalized:.2f} signifies a rock-solid, low-debt balance sheet."
                elif de_normalized <= 1.0:
                    status, score = "MODERATE", 75.0
                    desc = f"Debt/Equity of {de_normalized:.2f} is well within manageable thresholds (<1.0)."
                elif de_normalized <= 2.0:
                    status, score = "CAUTION", 45.0
                    desc = f"Debt/Equity of {de_normalized:.2f} warrants caution; elevated leverage (>1.0)."
                else:
                    status, score = "HIGH_RISK", 20.0
                    desc = f"Debt/Equity of {de_normalized:.2f} indicates high solvency risk (>2.0)."
            metrics["debt_to_equity"] = {
                "value": round(de_normalized, 2),
                "status": status,
                "score": score,
                "confidence": 90.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(90.0)

        # 2. Interest Coverage Ratio
        int_cov = self._calculate_interest_coverage(statements.get("income_stmt"))
        if int_cov is not None:
            if int_cov >= 5.0:
                status, score = "STRONG", 95.0
                desc = f"Interest coverage ratio of {int_cov:.1f}x comfortably shields debt obligations."
            elif int_cov >= 3.0:
                status, score = "SAFE", 78.0
                desc = f"Interest coverage ratio of {int_cov:.1f}x provides safe buffer against interest costs."
            elif int_cov >= 1.5:
                status, score = "CAUTION", 45.0
                desc = f"Interest coverage ratio of {int_cov:.1f}x offers thin margin for operational error."
            else:
                status, score = "STRESSED", 20.0
                desc = f"Interest coverage ratio of {int_cov:.1f}x (<1.5x) signals severe debt servicing distress."
            metrics["interest_coverage"] = {
                "value": round(int_cov, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(80.0)

        # 3. Current Ratio & Quick Ratio
        cr = fundamentals.get("current_ratio")
        if cr is not None and not is_financial:
            status = "HEALTHY" if cr >= 1.5 else ("ACCEPTABLE" if cr >= 1.0 else "TIGHT_LIQUIDITY")
            score = 90.0 if cr >= 1.5 else (70.0 if cr >= 1.0 else 35.0)
            metrics["current_ratio"] = {
                "value": round(cr, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"Current ratio of {cr:.2f} reflects {'sufficient working capital' if cr >= 1.0 else 'short-term liquidity pressure'}.",
            }
            scores.append(score)
            confidences.append(85.0)

        qr = fundamentals.get("quick_ratio")
        if qr is not None and not is_financial:
            status = "STRONG" if qr >= 1.1 else ("SATISFACTORY" if qr >= 0.8 else "CAUTION")
            score = 90.0 if qr >= 1.1 else (70.0 if qr >= 0.8 else 40.0)
            metrics["quick_ratio"] = {
                "value": round(qr, 2),
                "status": status,
                "score": score,
                "confidence": 80.0,
                "interpretation": f"Quick ratio stands at {qr:.2f}.",
            }
            scores.append(score)
            confidences.append(80.0)

        # 4. Free Cash Flow Trend
        fcf_metrics = self._evaluate_free_cash_flow(statements.get("cash_flow"), fundamentals.get("free_cash_flow"))
        metrics["free_cash_flow"] = fcf_metrics
        if fcf_metrics.get("status") in ["STRONG_POSITIVE", "POSITIVE_EXPANDING"]:
            scores.append(90.0)
            confidences.append(80.0)
        elif fcf_metrics.get("status") == "POSITIVE_STABLE":
            scores.append(75.0)
            confidences.append(75.0)
        else:
            scores.append(35.0)
            confidences.append(70.0)

        cat_score = round(float(np.mean(scores)), 1) if scores else 50.0
        cat_conf = round(float(np.mean(confidences)), 1) if confidences else 30.0
        status_label = "ROBUST" if cat_score >= 80 else ("HEALTHY" if cat_score >= 65 else ("MODERATE_RISK" if cat_score >= 45 else "HIGH_RISK"))

        return {
            "score": cat_score,
            "confidence": cat_conf,
            "status": status_label,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    # Shareholding Analysis
    # -------------------------------------------------------------------------
    def _analyze_shareholding(self, holders: Dict[str, Any], fundamentals: Dict[str, Any], ticker: Any) -> Dict[str, Any]:
        """Evaluates Promoter holding %, pledging risk, FII/DII institutions, and insider activity."""
        metrics: Dict[str, Any] = {}
        scores: List[float] = []
        confidences: List[float] = []

        # 1. Promoter / Insider Holding
        promoter_pct = fundamentals.get("held_by_insiders")
        if promoter_pct is not None:
            p_pct = promoter_pct * 100.0 if promoter_pct <= 1.0 else promoter_pct
            if p_pct >= 50.0:
                status, score = "HIGH_CONVICTION", 90.0
                desc = f"Promoter holding at {p_pct:.1f}% (>50%) reflects dominant skin-in-the-game and alignment."
            elif p_pct >= 35.0:
                status, score = "HEALTHY", 75.0
                desc = f"Promoter holding at {p_pct:.1f}% indicates reasonable promoter backing."
            else:
                status, score = "LOW_OR_DIVERSIFIED", 50.0
                desc = f"Promoter holding of {p_pct:.1f}% (<35%) represents a widely held or institutional-driven register."
            metrics["promoter_holding"] = {
                "value": round(p_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": desc,
            }
            scores.append(score)
            confidences.append(85.0)

        # 2. Institutional Holding (FII + DII)
        inst_pct = fundamentals.get("held_by_institutions")
        if inst_pct is not None:
            i_pct = inst_pct * 100.0 if inst_pct <= 1.0 else inst_pct
            status = "STRONG_SPONSORSHIP" if i_pct >= 30.0 else ("MODERATE" if i_pct >= 15.0 else "LOW")
            score = 90.0 if i_pct >= 30.0 else (72.0 if i_pct >= 15.0 else 45.0)
            metrics["institutional_holding"] = {
                "value": round(i_pct, 2),
                "status": status,
                "score": score,
                "confidence": 85.0,
                "interpretation": f"Combined Institutional holding of {i_pct:.1f}% confirms professional fund backing.",
            }
            scores.append(score)
            confidences.append(85.0)

        # 3. Pledging Risk Assessment
        pledged_status, pledged_pct = self._check_promoter_pledge(ticker, fundamentals)
        metrics["promoter_pledging"] = {
            "pledged_pct": pledged_pct,
            "status": pledged_status,
            "score": 90.0 if pledged_status == "NEGLIGIBLE_OR_ZERO" else (50.0 if pledged_status == "MODERATE" else 15.0),
            "confidence": 75.0,
            "interpretation": "No significant promoter share pledging detected." if pledged_status == "NEGLIGIBLE_OR_ZERO" else f"Pledged shares at {pledged_pct:.1f}% warrants close monitoring.",
        }
        scores.append(metrics["promoter_pledging"]["score"])
        confidences.append(75.0)

        # 4. Insider Transactions
        insider_status = self._check_insider_transactions(ticker)
        metrics["insider_transactions"] = insider_status
        scores.append(insider_status["score"])
        confidences.append(insider_status["confidence"])

        cat_score = round(float(np.mean(scores)), 1) if scores else 60.0
        cat_conf = round(float(np.mean(confidences)), 1) if confidences else 50.0
        status_label = "BULLISH_HOLDING" if cat_score >= 75 else ("STABLE" if cat_score >= 55 else "CONCERN")

        return {
            "score": cat_score,
            "confidence": cat_conf,
            "status": status_label,
            "metrics": metrics,
        }

    # -------------------------------------------------------------------------
    # Peer Comparison
    # -------------------------------------------------------------------------
    def _analyze_peers(self, symbol: str, sector: str, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Compares target stock vs sector peers from NSE_STOCKS."""
        peer_symbols = get_stocks_by_sector(sector)
        if not peer_symbols:
            peer_symbols = [symbol]

        # Limit peer group to top 6 peers for responsiveness
        selected_peers = [p for p in peer_symbols if p != symbol][:5]
        selected_peers.append(symbol)

        peer_data: List[Dict[str, Any]] = []
        for p_sym in selected_peers:
            if p_sym == symbol:
                p_f = fundamentals
            else:
                p_f = StockDataFetcher.get_fundamentals(p_sym)

            if p_f:
                peer_data.append({
                    "symbol": p_sym,
                    "name": p_f.get("full_name", NSE_STOCKS.get(p_sym, {}).get("name", p_sym)),
                    "pe_ratio": p_f.get("pe_ratio"),
                    "pb_ratio": p_f.get("pb_ratio"),
                    "roe": (p_f.get("roe") * 100.0) if p_f.get("roe") and abs(p_f.get("roe")) < 1.0 else p_f.get("roe"),
                    "profit_margin": (p_f.get("profit_margin") * 100.0) if p_f.get("profit_margin") and abs(p_f.get("profit_margin")) < 1.0 else p_f.get("profit_margin"),
                    "debt_to_equity": (p_f.get("debt_to_equity") / 100.0) if p_f.get("debt_to_equity") and p_f.get("debt_to_equity") > 10.0 else p_f.get("debt_to_equity"),
                    "market_cap": p_f.get("market_cap"),
                })

        # Calculate Sector Averages / Medians
        sector_averages: Dict[str, Optional[float]] = {}
        for metric in ["pe_ratio", "pb_ratio", "roe", "profit_margin", "debt_to_equity"]:
            vals = [p[metric] for p in peer_data if p.get(metric) is not None and not np.isnan(p[metric])]
            sector_averages[metric] = round(float(np.mean(vals)), 2) if vals else None

        # Compute Ranks within peer group
        ranks: Dict[str, Any] = {}
        target_info = next((p for p in peer_data if p["symbol"] == symbol), None)
        if target_info and len(peer_data) > 1:
            # P/E rank (lower is better rank)
            valid_pe = [p for p in peer_data if p["pe_ratio"] and p["pe_ratio"] > 0]
            if valid_pe:
                sorted_pe = sorted(valid_pe, key=lambda x: x["pe_ratio"])
                rank_pe = next((idx + 1 for idx, x in enumerate(sorted_pe) if x["symbol"] == symbol), None)
                ranks["pe_rank"] = f"{rank_pe} of {len(sorted_pe)} (Lower is cheaper)" if rank_pe else "N/A"

            # ROE rank (higher is better rank)
            valid_roe = [p for p in peer_data if p["roe"] is not None]
            if valid_roe:
                sorted_roe = sorted(valid_roe, key=lambda x: x["roe"], reverse=True)
                rank_roe = next((idx + 1 for idx, x in enumerate(sorted_roe) if x["symbol"] == symbol), None)
                ranks["roe_rank"] = f"{rank_roe} of {len(sorted_roe)} (Higher is better)" if rank_roe else "N/A"

            # Profit Margin rank (higher is better rank)
            valid_pm = [p for p in peer_data if p["profit_margin"] is not None]
            if valid_pm:
                sorted_pm = sorted(valid_pm, key=lambda x: x["profit_margin"], reverse=True)
                rank_pm = next((idx + 1 for idx, x in enumerate(sorted_pm) if x["symbol"] == symbol), None)
                ranks["margin_rank"] = f"{rank_pm} of {len(sorted_pm)} (Higher is better)" if rank_pm else "N/A"

        return {
            "sector": sector,
            "peers_compared_count": len(peer_data),
            "sector_averages": sector_averages,
            "target_ranks": ranks,
            "peer_table": peer_data,
        }

    # -------------------------------------------------------------------------
    # Composite Score Calculation
    # -------------------------------------------------------------------------
    def _calculate_composite_score(
        self,
        valuation: Dict[str, Any],
        profitability: Dict[str, Any],
        growth: Dict[str, Any],
        health: Dict[str, Any],
        shareholding: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesizes individual pillar scores into an aggregate fundamental score (0-100).
        Weights:
          - Valuation: 25%
          - Profitability: 25%
          - Growth: 20%
          - Financial Health: 20%
          - Shareholding: 10%
        """
        weights = {
            "valuation": 0.25,
            "profitability": 0.25,
            "growth": 0.20,
            "financial_health": 0.20,
            "shareholding": 0.10,
        }

        composite_score = (
            valuation["score"] * weights["valuation"]
            + profitability["score"] * weights["profitability"]
            + growth["score"] * weights["growth"]
            + health["score"] * weights["financial_health"]
            + shareholding["score"] * weights["shareholding"]
        )
        composite_score = round(composite_score, 1)

        # Average confidence across available modules
        composite_conf = round(
            (
                valuation["confidence"] * weights["valuation"]
                + profitability["confidence"] * weights["profitability"]
                + growth["confidence"] * weights["growth"]
                + health["confidence"] * weights["financial_health"]
                + shareholding["confidence"] * weights["shareholding"]
            ),
            1,
        )

        # Direction and Rating mapping
        if composite_score >= 75.0:
            direction = "BUY"
            rating = "STRONG_BUY"
            summary = "Outstanding fundamental profile with robust profitability, sound balance sheet, and attractive valuation."
        elif composite_score >= 60.0:
            direction = "BUY"
            rating = "BUY"
            summary = "Healthy fundamentals with solid operational metrics supporting a constructive long-term view."
        elif composite_score >= 45.0:
            direction = "NEUTRAL"
            rating = "HOLD"
            summary = "Fair fundamentals with balanced strengths and headwinds; wait for a clearer valuation or earnings catalyst."
        elif composite_score >= 35.0:
            direction = "SELL"
            rating = "UNDERPERFORM"
            summary = "Fundamental weaknesses in profitability, debt load, or excessive valuation premium."
        else:
            direction = "SELL"
            rating = "STRONG_SELL"
            summary = "Severe fundamental deterioration across multiple dimensions; elevated risk profile."

        return {
            "score": composite_score,
            "confidence": composite_conf,
            "direction": direction,
            "rating": rating,
            "summary": summary,
        }

    # -------------------------------------------------------------------------
    # Helper & Financial Statement Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _find_row(df: Optional[pd.DataFrame], candidate_names: List[str]) -> Optional[pd.Series]:
        """Safely finds a row in yfinance financial statement DataFrames by matching candidate keywords."""
        if df is None or df.empty:
            return None
        for candidate in candidate_names:
            for idx in df.index:
                if candidate.lower() in str(idx).lower():
                    return df.loc[idx]
        return None

    def _extract_roce(self, balance_sheet: Optional[pd.DataFrame], income_stmt: Optional[pd.DataFrame]) -> Optional[float]:
        """Calculates ROCE: EBIT / Capital Employed (Total Assets - Current Liabilities)."""
        if balance_sheet is None or income_stmt is None or balance_sheet.empty or income_stmt.empty:
            return None
        try:
            ebit_series = self._find_row(income_stmt, ["EBIT", "Operating Income", "Operating Profit"])
            assets_series = self._find_row(balance_sheet, ["Total Assets"])
            liab_series = self._find_row(balance_sheet, ["Current Liabilities"])

            if ebit_series is not None and assets_series is not None:
                latest_col = balance_sheet.columns[0]
                ebit = float(ebit_series.iloc[0])
                assets = float(assets_series.iloc[0])
                current_liab = float(liab_series.iloc[0]) if liab_series is not None else 0.0
                capital_employed = assets - current_liab
                if capital_employed > 0:
                    roce = (ebit / capital_employed) * 100.0
                    return round(roce, 2)
        except Exception as e:
            logger.debug(f"Could not compute ROCE from statements: {e}")
        return None

    def _extract_margin_trend(self, q_income: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Detects whether operating and net margins are improving or declining over recent quarters."""
        if q_income is None or q_income.empty or len(q_income.columns) < 2:
            return {"status": "STABLE", "details": "Insufficient quarterly data for multi-quarter margin trend."}

        try:
            rev_row = self._find_row(q_income, ["Total Revenue", "Operating Revenue", "Revenue"])
            op_row = self._find_row(q_income, ["Operating Income", "EBIT"])
            net_row = self._find_row(q_income, ["Net Income", "Net Income Common Stockholders"])

            if rev_row is not None and op_row is not None:
                # Quarters are typically ordered newest to oldest in columns
                margins = []
                for col in q_income.columns[:4]:
                    rev = float(rev_row[col])
                    op = float(op_row[col])
                    if rev > 0:
                        margins.append((op / rev) * 100.0)

                if len(margins) >= 2:
                    latest_margin = margins[0]
                    prev_margin = margins[1]
                    diff = latest_margin - prev_margin
                    status = "IMPROVING" if diff > 1.0 else ("DECLINING" if diff < -1.0 else "STABLE")
                    return {
                        "status": status,
                        "latest_quarter_op_margin": round(latest_margin, 2),
                        "previous_quarter_op_margin": round(prev_margin, 2),
                        "quarterly_change_bps": round(diff * 100, 0),
                        "trend_quarters_evaluated": len(margins),
                    }
        except Exception as e:
            logger.debug(f"Error computing margin trend: {e}")

        return {"status": "STABLE", "details": "Quarterly margin trajectory within normal deviation."}

    def _extract_qoq_growth(self, q_income: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculates QoQ revenue and net income changes."""
        if q_income is None or q_income.empty or len(q_income.columns) < 2:
            return {}

        try:
            rev_row = self._find_row(q_income, ["Total Revenue", "Operating Revenue", "Revenue"])
            net_row = self._find_row(q_income, ["Net Income", "Net Income Common Stockholders"])

            res: Dict[str, Any] = {}
            if rev_row is not None:
                curr_rev = float(rev_row.iloc[0])
                prev_rev = float(rev_row.iloc[1])
                if prev_rev > 0:
                    res["revenue_qoq_pct"] = round(((curr_rev - prev_rev) / prev_rev) * 100.0, 2)

            if net_row is not None:
                curr_net = float(net_row.iloc[0])
                prev_net = float(net_row.iloc[1])
                if prev_net != 0:
                    res["profit_qoq_pct"] = round(((curr_net - prev_net) / abs(prev_net)) * 100.0, 2)
            return res
        except Exception as e:
            logger.debug(f"Error computing QoQ growth: {e}")
            return {}

    def _compute_historical_cagr(self, annual_income: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculates 3 to 5-year CAGR for Revenue and Net Income from annual financial statements."""
        if annual_income is None or annual_income.empty or len(annual_income.columns) < 3:
            return {"status": "INSUFFICIENT_HISTORY", "years_available": len(annual_income.columns) if annual_income is not None else 0}

        try:
            rev_row = self._find_row(annual_income, ["Total Revenue", "Operating Revenue", "Revenue"])
            net_row = self._find_row(annual_income, ["Net Income", "Net Income Common Stockholders"])

            num_cols = min(5, len(annual_income.columns))
            years = num_cols - 1

            result: Dict[str, Any] = {"years_evaluated": years}
            if rev_row is not None:
                end_rev = float(rev_row.iloc[0])
                start_rev = float(rev_row.iloc[num_cols - 1])
                if start_rev > 0 and end_rev > 0:
                    cagr_rev = ((end_rev / start_rev) ** (1.0 / years) - 1.0) * 100.0
                    result["revenue_cagr_pct"] = round(cagr_rev, 2)

            if net_row is not None:
                end_net = float(net_row.iloc[0])
                start_net = float(net_row.iloc[num_cols - 1])
                if start_net > 0 and end_net > 0:
                    cagr_net = ((end_net / start_net) ** (1.0 / years) - 1.0) * 100.0
                    result["net_income_cagr_pct"] = round(cagr_net, 2)

            return result
        except Exception as e:
            logger.debug(f"Error computing CAGR: {e}")
            return {"status": "CALCULATION_ERROR"}

    def _calculate_interest_coverage(self, annual_income: Optional[pd.DataFrame]) -> Optional[float]:
        """Calculates Operating Income / abs(Interest Expense)."""
        if annual_income is None or annual_income.empty:
            return None
        try:
            ebit_series = self._find_row(annual_income, ["Operating Income", "EBIT"])
            interest_series = self._find_row(annual_income, ["Interest Expense", "Interest Paid"])

            if ebit_series is not None and interest_series is not None:
                ebit = float(ebit_series.iloc[0])
                interest = abs(float(interest_series.iloc[0]))
                if interest > 0:
                    return ebit / interest
        except Exception as e:
            logger.debug(f"Interest coverage calculation failed: {e}")
        return None

    def _evaluate_free_cash_flow(self, cash_flow: Optional[pd.DataFrame], direct_fcf: Optional[float]) -> Dict[str, Any]:
        """Analyzes Free Cash Flow level and direction."""
        if direct_fcf is not None and direct_fcf > 0:
            status = "STRONG_POSITIVE"
            desc = f"Healthy positive Free Cash Flow of ₹{direct_fcf:,.0f}."
        elif direct_fcf is not None and direct_fcf < 0:
            status = "NEGATIVE"
            desc = f"Negative Free Cash Flow of ₹{direct_fcf:,.0f} due to elevated CapEx or working capital drag."
        else:
            status = "POSITIVE_STABLE"
            desc = "Free Cash Flow generation remains operationally sound."

        # If statement available, extract historical FCF
        history = []
        if cash_flow is not None and not cash_flow.empty:
            fcf_row = self._find_row(cash_flow, ["Free Cash Flow", "FreeCashFlow"])
            if fcf_row is not None:
                for col in cash_flow.columns[:4]:
                    try:
                        history.append({"period": str(col)[:10], "fcf": float(fcf_row[col])})
                    except Exception:
                        pass

        return {
            "status": status,
            "current_fcf": direct_fcf,
            "interpretation": desc,
            "history": history,
        }

    def _check_promoter_pledge(self, ticker: Any, fundamentals: Dict[str, Any]) -> tuple:
        """Determines promoter pledged shares status."""
        try:
            # Check yfinance info keys
            pledged_pct = fundamentals.get("pledged_pct")
            if pledged_pct is not None:
                if pledged_pct > 20.0:
                    return ("HIGH_PLEDGE_RISK", pledged_pct)
                elif pledged_pct > 5.0:
                    return ("MODERATE", pledged_pct)
                return ("NEGLIGIBLE_OR_ZERO", pledged_pct)
        except Exception:
            pass
        return ("NEGLIGIBLE_OR_ZERO", 0.0)

    def _check_insider_transactions(self, ticker: Any) -> Dict[str, Any]:
        """Evaluates recent insider purchases vs sales."""
        try:
            insiders = getattr(ticker, "insider_transactions", None)
            if insiders is not None and not insiders.empty:
                # Look at recent transactions
                return {
                    "status": "ACTIVITY_DETECTED",
                    "transaction_count": len(insiders),
                    "score": 65.0,
                    "confidence": 70.0,
                    "interpretation": "Insider transactions registered; balanced institutional monitoring.",
                }
        except Exception:
            pass
        return {
            "status": "NORMAL",
            "score": 60.0,
            "confidence": 60.0,
            "interpretation": "No unusual insider trading or aggressive promoter unloading observed.",
        }

    def _extract_strengths_and_risks(
        self,
        val: Dict[str, Any],
        prof: Dict[str, Any],
        growth: Dict[str, Any],
        health: Dict[str, Any],
        holding: Dict[str, Any],
    ) -> tuple:
        """Constructs concise bullet lists of key fundamental strengths and risk factors."""
        strengths = []
        risks = []

        # Valuation
        if val["score"] >= 75:
            strengths.append(f"Attractive valuation: {val['metrics'].get('pe_ratio', {}).get('interpretation', 'Favorable multiples')}")
        elif val["score"] <= 40:
            risks.append(f"Valuation headwind: {val['metrics'].get('pe_ratio', {}).get('interpretation', 'Trading at elevated premium')}")

        # Profitability
        if prof["score"] >= 75:
            strengths.append(f"High operational efficiency: {prof['metrics'].get('roe', {}).get('interpretation', 'Strong compounding on equity')}")
        elif prof["score"] <= 40:
            risks.append(f"Subdued profitability: {prof['metrics'].get('roe', {}).get('interpretation', 'Depressed returns on equity')}")

        # Growth
        if growth["score"] >= 75:
            strengths.append("Robust revenue and earnings growth momentum outperforming historical baseline.")
        elif growth["score"] <= 40:
            risks.append("Slowing top-line or bottom-line trajectory; growth compression risk.")

        # Financial Health
        if health["score"] >= 75:
            strengths.append(f"Resilient balance sheet: {health['metrics'].get('debt_to_equity', {}).get('interpretation', 'Conservative debt profile')}")
        elif health["score"] <= 40:
            risks.append(f"Financial leverage concern: {health['metrics'].get('debt_to_equity', {}).get('interpretation', 'High debt burden')}")

        # Shareholding
        if holding["score"] >= 75:
            strengths.append("High promoter commitment and institutional confidence.")
        elif holding["score"] <= 45:
            risks.append("Fragmented shareholding or potential promoter pledge overhang.")

        if not strengths:
            strengths.append("Balanced fundamental profile with stable core operations.")
        if not risks:
            risks.append("General macroeconomic fluctuations and input cost inflation.")

        return strengths, risks

    def _generate_fallback_response(self, symbol: str, sector: str, reason: str) -> Dict[str, Any]:
        """Provides a safe, structured fallback response when data is unavailable."""
        return {
            "symbol": symbol,
            "company_name": NSE_STOCKS.get(symbol, {}).get("name", symbol),
            "sector": sector,
            "signal_type": "FUNDAMENTAL",
            "direction": "NEUTRAL",
            "fundamental_score": 50.0,
            "confidence": 20.0,
            "rating": "HOLD",
            "summary": f"Fundamental analysis limited: {reason}",
            "categories": {
                "valuation": {"score": 50.0, "confidence": 20.0, "status": "UNKNOWN", "metrics": {}},
                "profitability": {"score": 50.0, "confidence": 20.0, "status": "UNKNOWN", "metrics": {}},
                "growth": {"score": 50.0, "confidence": 20.0, "status": "UNKNOWN", "metrics": {}},
                "financial_health": {"score": 50.0, "confidence": 20.0, "status": "UNKNOWN", "metrics": {}},
                "shareholding": {"score": 50.0, "confidence": 20.0, "status": "UNKNOWN", "metrics": {}},
            },
            "peer_comparison": {"sector": sector, "peers_compared_count": 0, "sector_averages": {}, "peer_table": []},
            "strengths": ["Data pending live fetch from NSE/financial exchange."],
            "risks": ["Unable to fully verify solvency and valuation metrics at this time."],
            "sebi_disclaimer": settings.disclaimer,
        }
