import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Chart from '../components/Chart';
import IndicatorPanel from '../components/IndicatorPanel';
import TradePlanCard from '../components/TradePlanCard';
import PredictionCard from '../components/PredictionCard';
import Watchlist from '../components/Watchlist';
import Screener from '../components/Screener';
import ErrorBoundary from '../components/ErrorBoundary';
import {
  getQuote,
  getOHLCV,
  getFullAnalysis,
  getPrediction,
  getTradePlan,
} from '../api/client';

const POPULAR_STOCKS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'TATAMOTORS'];

export default function Dashboard() {
  const [symbol, setSymbol] = useState('RELIANCE');
  const [searchInput, setSearchInput] = useState('');
  const [period, setPeriod] = useState('1y');
  const [tradingMode, setTradingMode] = useState('SWING');

  // Progressive loading states — independent per card
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [planLoading, setPlanLoading] = useState(false);
  const [predictionLoading, setPredictionLoading] = useState(false);

  // Data states
  const [quoteData, setQuoteData] = useState(null);
  const [ohlcvData, setOhlcvData] = useState([]);
  const [analysisData, setAnalysisData] = useState(null);
  const [tradePlanData, setTradePlanData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);

  // Global & Card-specific errors
  const [error, setError] = useState(null);

  const abortControllerRef = useRef(null);
  const navigate = useNavigate();

  // Progressive decoupled loading
  const loadStockData = useCallback(
    (targetSymbol, currentMode = tradingMode) => {
      const sym = (targetSymbol || symbol).toUpperCase().trim();
      if (!sym) return;

      // 1. Cancel previous in-flight requests when switching symbols
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;
      const config = { signal: controller.signal };

      setError(null);
      setQuoteLoading(true);
      setChartLoading(true);
      setPlanLoading(true);
      setAnalysisLoading(true);
      setPredictionLoading(true);

      console.log(`[Dashboard] Progressive streaming initiated for ${sym} (Mode: ${currentMode})...`);

      // STAGE 1: Instant Quote (<200ms) — price, change %, stock name
      getQuote(sym, config)
        .then((res) => {
          if (res?.data) {
            setQuoteData(res.data);
          }
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.warn(`[Dashboard] Quote fetch error for ${sym}:`, err.message);
          }
        })
        .finally(() => {
          setQuoteLoading(false);
        });

      // STAGE 2: Chart OHLCV (<800ms) — TradingView candlesticks
      getOHLCV(sym, period, '1d', config)
        .then((res) => {
          const candles = Array.isArray(res?.data?.data) ? res.data.data : [];
          setOhlcvData(candles);
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.warn(`[Dashboard] OHLCV fetch error for ${sym}:`, err.message);
            setOhlcvData([]);
          }
        })
        .finally(() => {
          setChartLoading(false);
        });

      // STAGE 3A: Trade Plan with Selected Mode (~1-2s)
      getTradePlan(sym, currentMode, 1000000, 1.0, config)
        .then((res) => {
          if (res?.data) {
            setTradePlanData(res.data);
          }
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.warn(`[Dashboard] Trade plan error for ${sym}:`, err.message);
          }
        })
        .finally(() => {
          setPlanLoading(false);
        });

      // STAGE 3B: Technical & Full Analysis (~1-2s)
      getFullAnalysis(sym, period, config)
        .then((res) => {
          if (res?.data) {
            setAnalysisData(res.data);
          }
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.warn(`[Dashboard] Analysis error for ${sym}:`, err.message);
          }
        })
        .finally(() => {
          setAnalysisLoading(false);
        });

      // STAGE 4: ML Ensemble Prediction (~2-3s)
      getPrediction(sym, config)
        .then((res) => {
          if (res?.data) {
            setPredictionData(res.data);
          }
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.warn(`[Dashboard] Prediction error for ${sym}:`, err.message);
          }
        })
        .finally(() => {
          setPredictionLoading(false);
        });
    },
    [symbol, period, tradingMode]
  );

  useEffect(() => {
    loadStockData(symbol, tradingMode);
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [symbol, period, tradingMode, loadStockData]);

  // Debounced search submit
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const clean = searchInput.trim().toUpperCase();
    if (clean) {
      setSymbol(clean);
      setSearchInput('');
    }
  };

  // Derive display values from quoteData, analysisData, or latest candle
  const currentPrice =
    quoteData?.current_price ??
    analysisData?.current_price ??
    (ohlcvData.length > 0 ? ohlcvData[ohlcvData.length - 1]?.close : null);

  const changePct = quoteData?.change_pct ?? analysisData?.change_pct ?? null;
  const companyName = quoteData?.name ?? analysisData?.fundamental?.info?.name ?? 'NSE Equity';
  const sector = quoteData?.sector ?? analysisData?.fundamental?.sector ?? 'Diversified';
  const verdict = analysisData?.overall_verdict ?? null;
  const confidence = analysisData?.overall_confidence ?? 50;

  return (
    <div className="space-y-6">
      {/* Search and Period Bar */}
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 card p-4 shadow-md">
        <form onSubmit={handleSearchSubmit} className="flex gap-2 w-full md:w-auto">
          <input
            type="text"
            placeholder="Search NSE stock (e.g. RELIANCE, TCS)..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm w-full md:w-72 focus:outline-none focus:border-blue-500 text-gray-100 placeholder-gray-500"
          />
          <button
            type="submit"
            className="btn-primary text-sm whitespace-nowrap flex items-center gap-1.5"
          >
            <span>🔍</span> Analyze
          </button>
        </form>

        <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0">
          <span className="text-xs text-gray-500 font-medium">Popular:</span>
          {POPULAR_STOCKS.map((stk) => (
            <button
              key={stk}
              onClick={() => setSymbol(stk)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors font-medium ${
                symbol === stk
                  ? 'bg-blue-600 text-white font-semibold shadow'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {stk}
            </button>
          ))}
        </div>

        {/* Mode Switcher */}
        <div className="flex items-center gap-1 bg-gray-950/80 border border-gray-700/80 p-1 rounded-lg">
          <button
            onClick={() => setTradingMode('INTRADAY')}
            className={`px-2.5 py-1 text-xs rounded font-semibold transition-all ${
              tradingMode === 'INTRADAY'
                ? 'bg-blue-600 text-white shadow'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            ⚡ Intraday
          </button>
          <button
            onClick={() => setTradingMode('SWING')}
            className={`px-2.5 py-1 text-xs rounded font-semibold transition-all ${
              tradingMode === 'SWING'
                ? 'bg-blue-600 text-white shadow'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            🎯 Swing (2-5d)
          </button>
          <button
            onClick={() => setTradingMode('SHORT_TERM')}
            className={`px-2.5 py-1 text-xs rounded font-semibold transition-all ${
              tradingMode === 'SHORT_TERM'
                ? 'bg-blue-600 text-white shadow'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            🚀 Short-term (1-3w)
          </button>
        </div>

        <div className="flex items-center gap-1">
          {['3mo', '6mo', '1y', '2y'].map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-2.5 py-1 text-xs rounded font-medium transition-colors ${
                period === p
                  ? 'bg-gray-700 text-blue-400 font-bold'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {p.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Global Error Banner */}
      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800 text-red-200 rounded-xl text-sm flex items-center justify-between gap-4">
          <div>
            <p className="font-semibold mb-1">⚠️ Notice</p>
            <p className="text-xs text-red-300">{error}</p>
          </div>
          <button
            onClick={() => loadStockData(symbol)}
            className="px-3 py-1.5 bg-red-800 hover:bg-red-700 text-white rounded-lg text-xs font-semibold whitespace-nowrap"
          >
            Retry Fetch
          </button>
        </div>
      )}

      {/* Header Price & Verdict Card */}
      <ErrorBoundary name="Stock Header" onReset={() => loadStockData(symbol)}>
        <div className="card flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-md">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold tracking-tight text-white">{symbol}</h1>
              <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400 font-mono">
                NSE
              </span>

              {/* Signal Verdict */}
              {analysisLoading ? (
                <span className="badge-neutral text-xs animate-pulse">
                  🔄 Calculating Signals...
                </span>
              ) : verdict ? (
                <span
                  className={
                    verdict === 'BULLISH'
                      ? 'badge-bull text-xs font-semibold'
                      : verdict === 'BEARISH'
                      ? 'badge-bear text-xs font-semibold'
                      : 'badge-neutral text-xs font-semibold'
                  }
                >
                  {verdict} ({confidence}% Confidence)
                </span>
              ) : (
                <span className="badge-neutral text-xs text-gray-400">
                  Setup Evaluated
                </span>
              )}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {companyName} • Sector: {sector}
            </p>
          </div>

          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-2xl font-mono font-bold text-white">
                {quoteLoading && currentPrice == null ? (
                  <span className="text-gray-500 text-lg animate-pulse">₹...</span>
                ) : currentPrice != null ? (
                  `₹${Number(currentPrice).toFixed(2)}`
                ) : (
                  '₹—'
                )}
              </div>
              {changePct != null && (
                <div
                  className={`text-sm font-semibold font-mono ${
                    Number(changePct) >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}
                >
                  {Number(changePct) >= 0 ? '+' : ''}
                  {Number(changePct).toFixed(2)}%
                </div>
              )}
            </div>
            <button
              onClick={() => navigate(`/analysis?symbol=${symbol}`)}
              className="btn-secondary text-xs flex items-center gap-1"
            >
              Deep Dive <span>→</span>
            </button>
          </div>
        </div>
      </ErrorBoundary>

      {/* Main Grid: Chart + Indicator Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Interactive Chart & Trade Plan */}
        <div className="lg:col-span-2 space-y-6">
          <ErrorBoundary name="Interactive Chart" onReset={() => loadStockData(symbol)}>
            <Chart
              data={ohlcvData}
              tradePlan={tradePlanData}
              title={`${symbol} • ${period.toUpperCase()} Candlestick Chart with Volume & Trade Levels`}
              height={480}
              loading={chartLoading && ohlcvData.length === 0}
            />
          </ErrorBoundary>

          <ErrorBoundary name="Trade Plan Card">
            {tradePlanData ? (
              <TradePlanCard plan={tradePlanData} />
            ) : planLoading ? (
              <div className="card p-6 text-center text-sm text-gray-400">
                <div className="spinner mx-auto mb-2" />
                <span className="text-xs">Computing confluence buy zones & stop-loss levels...</span>
              </div>
            ) : null}
          </ErrorBoundary>
        </div>

        {/* Right 1 Col: AI Prediction, Technical Indicators, Watchlist */}
        <div className="space-y-6">
          <ErrorBoundary name="ML Prediction Card">
            {predictionData ? (
              <PredictionCard data={predictionData} />
            ) : predictionLoading ? (
              <div className="card p-6 text-center text-sm text-gray-400">
                <div className="spinner mx-auto mb-2" />
                <span className="text-xs">Evaluating machine learning ensemble direction...</span>
              </div>
            ) : null}
          </ErrorBoundary>

          <ErrorBoundary name="Indicator Panel">
            {analysisData?.technical ? (
              <IndicatorPanel data={analysisData.technical} />
            ) : analysisLoading ? (
              <div className="card p-6 text-center text-sm text-gray-400">
                <div className="spinner mx-auto mb-2" />
                <span className="text-xs">Calculating momentum, trend & volume indicators...</span>
              </div>
            ) : null}
          </ErrorBoundary>

          <ErrorBoundary name="Watchlist Card">
            <Watchlist onSelectStock={(stk) => setSymbol(stk)} />
          </ErrorBoundary>
        </div>
      </div>

      {/* Bottom Section: Live Screener */}
      <div className="mt-8">
        <h2 className="text-lg font-bold mb-3 flex items-center gap-2 text-white">
          <span>⚡</span> Live Market Scanner & Opportunities
        </h2>
        <ErrorBoundary name="Live Screener">
          <Screener onSelectStock={(stk) => setSymbol(stk)} />
        </ErrorBoundary>
      </div>
    </div>
  );
}
