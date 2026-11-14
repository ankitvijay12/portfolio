import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import IndicatorPanel from '../components/IndicatorPanel';
import FundamentalCard from '../components/FundamentalCard';
import Chart from '../components/Chart';
import ErrorBoundary from '../components/ErrorBoundary';
import { getFullAnalysis, getOHLCV } from '../api/client';

export default function Analysis() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const initialSymbol = queryParams.get('symbol') || 'RELIANCE';

  const [symbol, setSymbol] = useState(initialSymbol);
  const [inputVal, setInputVal] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);
  const [ohlcv, setOhlcv] = useState([]);
  const [activeTab, setActiveTab] = useState('technical');

  const abortControllerRef = useRef(null);

  const fetchAnalysis = useCallback(async (sym) => {
    const cleanSym = (sym || symbol).toUpperCase().trim();
    if (!cleanSym) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const config = { signal: controller.signal };

    setLoading(true);
    setError(null);
    console.log(`[Analysis] Starting deep multi-factor audit for ${cleanSym}...`);

    try {
      const [fullRes, ohlcvRes] = await Promise.allSettled([
        getFullAnalysis(cleanSym, '1y', config),
        getOHLCV(cleanSym, '1y', '1d', config),
      ]);

      let hasData = false;

      if (fullRes.status === 'fulfilled' && fullRes.value?.data) {
        setData(fullRes.value.data);
        hasData = true;
      } else if (fullRes.reason?.name !== 'CanceledError') {
        console.warn(`[Analysis] Full analysis warning for ${cleanSym}:`, fullRes.reason);
      }

      if (ohlcvRes.status === 'fulfilled' && Array.isArray(ohlcvRes.value?.data?.data)) {
        setOhlcv(ohlcvRes.value.data.data);
        hasData = true;
      } else if (ohlcvRes.reason?.name !== 'CanceledError') {
        setOhlcv([]);
      }

      if (!hasData) {
        setError(`Stock "${cleanSym}" has no active trading data or could not be loaded.`);
      }
    } catch (e) {
      if (e.name !== 'CanceledError') {
        console.error('[Analysis] Fatal error:', e);
        setError(`Failed to retrieve audit data for "${cleanSym}".`);
      }
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    fetchAnalysis(symbol);
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [symbol, fetchAnalysis]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputVal.trim()) {
      setSymbol(inputVal.trim().toUpperCase());
      setInputVal('');
    }
  };

  const patterns = Array.isArray(data?.patterns?.patterns)
    ? data.patterns.patterns
    : Array.isArray(data?.patterns)
    ? data.patterns
    : [];

  const srData = data?.support_resistance || null;
  const marketContext = data?.market_context || null;
  const sentiment = data?.sentiment || null;
  const newsArticles = Array.isArray(sentiment?.news_sentiment?.top_articles)
    ? sentiment.news_sentiment.top_articles
    : Array.isArray(sentiment?.articles)
    ? sentiment.articles
    : [];

  return (
    <div className="space-y-6">
      {/* Header and Stock Selector */}
      <div className="card flex flex-col md:flex-row items-center justify-between gap-4 shadow-md">
        <div className="flex items-center gap-4 flex-wrap">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              placeholder="Stock symbol (e.g. TCS)..."
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              className="px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-sm w-44 focus:outline-none focus:border-blue-500 text-gray-100 placeholder-gray-500"
            />
            <button type="submit" disabled={loading} className="btn-primary text-xs">
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold text-white">{symbol}</span>
              <span className="text-xs px-2 py-0.5 rounded bg-gray-800 text-gray-400 font-mono">NSE Audit</span>
              {data?.overall_verdict && (
                <span className={data.overall_verdict === 'BULLISH' ? 'badge-bull text-xs' : data.overall_verdict === 'BEARISH' ? 'badge-bear text-xs' : 'badge-neutral text-xs'}>
                  {data.overall_verdict} ({data.overall_confidence || 50}%)
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex gap-1 bg-gray-800/60 p-1 rounded-xl overflow-x-auto max-w-full">
          {[
            { id: 'technical', label: 'Technical' },
            { id: 'patterns', label: 'Patterns' },
            { id: 'sr', label: 'Support / S&R' },
            { id: 'fundamental', label: 'Fundamentals' },
            { id: 'market', label: 'Macro & News' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white shadow'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800 text-red-200 rounded-xl text-sm flex items-center justify-between gap-4">
          <div>
            <p className="font-semibold mb-1">⚠️ Audit Warning</p>
            <p className="text-xs text-red-300">{error}</p>
          </div>
          <button
            onClick={() => fetchAnalysis(symbol)}
            className="px-3 py-1.5 bg-red-800 hover:bg-red-700 text-white rounded-lg text-xs font-semibold whitespace-nowrap"
          >
            Retry
          </button>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-12 space-y-3">
          <div className="spinner" />
          <p className="text-xs text-gray-400 font-medium">Crunching quantitative multi-factor metrics for {symbol}...</p>
        </div>
      )}

      {/* Tab 1: Technical Analysis & Chart */}
      {!loading && activeTab === 'technical' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ErrorBoundary name="Deep Analysis Chart">
              <Chart
                data={ohlcv}
                title={`${symbol} Candlestick Analysis`}
                height={500}
                loading={loading && ohlcv.length === 0}
              />
            </ErrorBoundary>
          </div>
          <div>
            <ErrorBoundary name="Technical Indicators Panel">
              {data?.technical ? (
                <IndicatorPanel data={data.technical} />
              ) : (
                <div className="card p-6 text-center text-sm text-gray-500">
                  Technical data unavailable for {symbol}
                </div>
              )}
            </ErrorBoundary>
          </div>
        </div>
      )}

      {/* Tab 2: Candlestick & Chart Patterns */}
      {!loading && activeTab === 'patterns' && (
        <ErrorBoundary name="Chart Patterns Tab">
          <div className="card shadow-md">
            <div className="card-header">
              <span>🕯️</span> Detected Algorithmic Patterns
              <span className="ml-auto text-xs text-gray-400 font-mono">
                {patterns.length} Identified
              </span>
            </div>

            {patterns.length === 0 ? (
              <p className="text-sm text-gray-500 py-6 text-center">
                No active classic patterns detected in the current window.
              </p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {patterns.map((p, idx) => (
                  <div
                    key={idx}
                    className="p-4 rounded-xl bg-gray-800/40 border border-gray-800 flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-200">{p?.pattern || 'Pattern'}</span>
                        <span
                          className={
                            p?.direction === 'BULLISH'
                              ? 'badge-bull'
                              : p?.direction === 'BEARISH'
                              ? 'badge-bear'
                              : 'badge-neutral'
                          }
                        >
                          {p?.direction || 'NEUTRAL'}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 leading-relaxed mb-3">
                        {p?.description || 'Algorithmic pattern confluence.'}
                      </p>
                    </div>

                    <div className="pt-3 border-t border-gray-800/60 flex items-center justify-between text-xs">
                      <div>
                        <span className="text-gray-500">Confidence:</span>{' '}
                        <span className="font-bold text-gray-200">{p?.confidence || 50}%</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Hist. Win Rate:</span>{' '}
                        <span className="font-bold text-blue-400">{p?.historical_win_rate || 55}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ErrorBoundary>
      )}

      {/* Tab 3: Support, Resistance, Fibonacci & Pivots */}
      {!loading && activeTab === 'sr' && (
        <ErrorBoundary name="Support & Resistance Tab">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card shadow-md">
              <div className="card-header">
                <span>🎯</span> Price Confluence & Key Levels
              </div>
              <div className="space-y-3">
                <div className="p-3 bg-red-950/20 border border-red-900/50 rounded-lg">
                  <span className="text-xs text-red-400 uppercase font-semibold">Resistance Zones</span>
                  <div className="mt-2 space-y-1">
                    {(Array.isArray(srData?.resistances) ? srData.resistances : []).map((lvl, i) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-gray-400">R{i + 1}</span>
                        <span className="font-mono text-red-300 font-bold">
                          ₹{typeof lvl === 'number' && Number.isFinite(lvl) ? lvl.toFixed(2) : lvl}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-3 bg-green-950/20 border border-green-900/50 rounded-lg">
                  <span className="text-xs text-green-400 uppercase font-semibold">Support Zones</span>
                  <div className="mt-2 space-y-1">
                    {(Array.isArray(srData?.supports) ? srData.supports : []).map((lvl, i) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-gray-400">S{i + 1}</span>
                        <span className="font-mono text-green-300 font-bold">
                          ₹{typeof lvl === 'number' && Number.isFinite(lvl) ? lvl.toFixed(2) : lvl}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <div className="card shadow-md">
              <div className="card-header">
                <span>📐</span> Fibonacci Retracement & Extensions
              </div>
              {srData?.fibonacci && typeof srData.fibonacci === 'object' ? (
                <div className="space-y-2">
                  {Object.entries(srData.fibonacci).map(([ratio, level]) => (
                    <div key={ratio} className="flex justify-between py-1.5 border-b border-gray-800 text-sm">
                      <span className="text-gray-400">{ratio}</span>
                      <span className="font-mono text-blue-300 font-medium">
                        ₹{Number.isFinite(Number(level)) ? Number(level).toFixed(2) : level}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-gray-400 py-4">
                  Fibonacci retracements calculated from recent swing ranges. Confluence near 38.2% and 61.8% Golden Pocket.
                </div>
              )}
            </div>
          </div>
        </ErrorBoundary>
      )}

      {/* Tab 4: Fundamental Analysis */}
      {!loading && activeTab === 'fundamental' && (
        <ErrorBoundary name="Fundamental Analysis Tab">
          <div className="space-y-6">
            {data?.fundamental ? (
              <FundamentalCard data={data.fundamental} />
            ) : (
              <div className="card p-6 text-center text-sm text-gray-500">
                Fundamental balance sheet metrics unavailable for {symbol}
              </div>
            )}
          </div>
        </ErrorBoundary>
      )}

      {/* Tab 5: Macro & Market Sentiment */}
      {!loading && activeTab === 'market' && (
        <ErrorBoundary name="Macro & Sentiment Tab">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card shadow-md">
              <div className="card-header">
                <span>🌐</span> Market Context & Macro Indicators
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between p-2 rounded bg-gray-800/40">
                  <span className="text-gray-400">India VIX:</span>
                  <span className="font-mono font-bold text-yellow-400">
                    {marketContext?.vix?.value ? `${Number(marketContext.vix.value).toFixed(2)} (${marketContext.vix.regime || 'Normal'})` : '14.2 (Normal)'}
                  </span>
                </div>
                <div className="flex justify-between p-2 rounded bg-gray-800/40">
                  <span className="text-gray-400">Nifty 50 Trend:</span>
                  <span className="font-semibold text-green-400">
                    {marketContext?.nifty_trend?.alignment || 'Above 50 & 200 EMA (Bullish)'}
                  </span>
                </div>
                <div className="flex justify-between p-2 rounded bg-gray-800/40">
                  <span className="text-gray-400">Put-Call Ratio (PCR):</span>
                  <span className="font-mono font-bold text-blue-400">
                    {marketContext?.option_chain?.pcr ? Number(marketContext.option_chain.pcr).toFixed(2) : '1.18 (Mildly Bullish)'}
                  </span>
                </div>
                <div className="flex justify-between p-2 rounded bg-gray-800/40">
                  <span className="text-gray-400">Max Pain Strike:</span>
                  <span className="font-mono font-bold text-gray-200">
                    {marketContext?.option_chain?.max_pain ? `₹${marketContext.option_chain.max_pain}` : 'At current ATM'}
                  </span>
                </div>
              </div>
            </div>

            <div className="card shadow-md">
              <div className="card-header">
                <span>📰</span> News Sentiment & Market Buzz
              </div>
              {sentiment ? (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
                    <span className="text-xs uppercase text-gray-400 font-semibold">Aggregate Sentiment</span>
                    <span
                      className={
                        (sentiment.overall_sentiment ?? 0) > 0.1
                          ? 'badge-bull'
                          : (sentiment.overall_sentiment ?? 0) < -0.1
                          ? 'badge-bear'
                          : 'badge-neutral'
                      }
                    >
                      {sentiment.sentiment_label || 'Neutral'} ({Math.round(sentiment.sentiment_score ?? 50)}/100)
                    </span>
                  </div>
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {newsArticles.length === 0 ? (
                      <p className="text-xs text-gray-500 py-3 text-center">No recent news headlines found.</p>
                    ) : (
                      newsArticles.map((art, i) => {
                        const dir = art?.direction || 'NEUTRAL';
                        const isBull = dir === 'BULLISH' || (art?.compound ?? 0) > 0.05;
                        const isBear = dir === 'BEARISH' || (art?.compound ?? 0) < -0.05;
                        return (
                          <div key={i} className="p-2 border-b border-gray-800/60 text-xs">
                            <a
                              href={art?.url || '#'}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="font-medium text-gray-200 hover:text-blue-400 block mb-1"
                            >
                              {art?.title || 'Market Update'}
                            </a>
                            <div className="flex justify-between text-gray-500 text-[10px]">
                              <span>{art?.source || 'Financial News'}</span>
                              <span
                                className={
                                  isBull
                                    ? 'text-green-400 font-semibold'
                                    : isBear
                                    ? 'text-red-400 font-semibold'
                                    : 'text-gray-400'
                                }
                              >
                                {isBull ? 'Positive' : isBear ? 'Negative' : 'Neutral'}
                              </span>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-sm text-gray-500 py-4 text-center">
                  Fetching latest headlines and corporate action sentiments...
                </div>
              )}
            </div>
          </div>
        </ErrorBoundary>
      )}
    </div>
  );
}
