import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  getShortTermScan,
  getShortTermBacktestStats,
  getTaxConfig,
  calculateTax,
  getJournal,
  addToJournal,
  updateJournal,
  deleteFromJournal,
} from '../api/client';
import ErrorBoundary from '../components/ErrorBoundary';

const SETUPS = [
  { code: 'ALL', label: 'All Setups' },
  { code: 'BREAKOUT', label: '20d Breakout' },
  { code: 'PULLBACK', label: 'EMA Pullback' },
  { code: 'OVERSOLD_BOUNCE', label: 'Oversold Bounce' },
  { code: 'ENGULFING_HAMMER', label: 'Engulfing / Hammer' },
  { code: 'BOLLINGER_SQUEEZE', label: 'Bollinger Squeeze' },
  { code: 'STRONG_RS', label: 'Strong Relative Strength' },
  { code: 'GAP_UP', label: 'Gap-Up Momentum' },
];

const SECTORS = [
  'All',
  'Technology',
  'Financial Services',
  'Energy',
  'Automobile',
  'Healthcare',
  'FMCG',
  'Materials',
  'Utilities',
  'Industrials',
  'Consumer Discretionary',
];

export default function ShortTermPage() {
  const [activeTab, setActiveTab] = useState('scanner'); // 'scanner', 'journal', 'tax', 'backtest'
  const [mode, setMode] = useState('SWING'); // 'INTRADAY', 'SWING', 'SHORT_TERM'
  const [selectedSetup, setSelectedSetup] = useState('ALL');
  const [selectedSector, setSelectedSector] = useState('All');
  const [minConfidence, setMinConfidence] = useState(50);

  // Scanner state
  const [scans, setScans] = useState([]);
  const [scannerLoading, setScannerLoading] = useState(false);
  const [scannerError, setScannerError] = useState(null);
  const [lastRefreshed, setLastRefreshed] = useState(null);

  // Journal state
  const [journalData, setJournalData] = useState({ summary: {}, trades: [] });
  const [journalLoading, setJournalLoading] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  // Tax state
  const [taxConfig, setTaxConfig] = useState(null);
  const [taxInputs, setTaxInputs] = useState({
    buy_price: 1500,
    sell_price: 1850,
    quantity: 100,
    buy_date: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    sell_date: new Date().toISOString().split('T')[0],
    trade_type: 'DELIVERY',
    brokerage_per_order: 20.0,
  });
  const [taxResult, setTaxResult] = useState(null);
  const [taxLoading, setTaxLoading] = useState(false);

  // Backtest stats state
  const [backtestStats, setBacktestStats] = useState(null);
  const [backtestLoading, setBacktestLoading] = useState(false);

  const abortControllerRef = useRef(null);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // 1. Fetch Scanner Setups
  const fetchScans = useCallback(
    (forceRefresh = false) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setScannerLoading(true);
      setScannerError(null);

      const params = {
        mode,
        min_confidence: minConfidence,
        force_refresh: forceRefresh,
      };
      if (selectedSetup !== 'ALL') params.setup = selectedSetup;
      if (selectedSector !== 'All') params.sector = selectedSector;

      getShortTermScan(params, { signal: controller.signal })
        .then((res) => {
          if (res?.data?.data) {
            setScans(res.data.data);
            setLastRefreshed(new Date().toLocaleTimeString());
          }
        })
        .catch((err) => {
          if (err.name !== 'CanceledError' && err.message !== 'canceled') {
            console.error('[ShortTerm] Scanner error:', err);
            setScannerError(err.response?.data?.detail || err.message || 'Failed to fetch scan results');
          }
        })
        .finally(() => {
          setScannerLoading(false);
        });
    },
    [mode, selectedSetup, selectedSector, minConfidence]
  );

  // 2. Fetch Journal
  const fetchJournal = useCallback(() => {
    setJournalLoading(true);
    getJournal()
      .then((res) => {
        if (res?.data) {
          setJournalData(res.data);
        }
      })
      .catch((err) => console.error('[ShortTerm] Journal fetch error:', err))
      .finally(() => setJournalLoading(false));
  }, []);

  // 3. Fetch Tax Config & Calculate
  const fetchTaxInfo = useCallback(() => {
    setTaxLoading(true);
    getTaxConfig()
      .then((res) => {
        if (res?.data) setTaxConfig(res.data);
      })
      .catch((err) => console.error('[ShortTerm] Tax config error:', err));

    calculateTax(taxInputs)
      .then((res) => {
        if (res?.data) setTaxResult(res.data);
      })
      .catch((err) => console.error('[ShortTerm] Tax calculation error:', err))
      .finally(() => setTaxLoading(false));
  }, [taxInputs]);

  // 4. Fetch Backtest Stats
  const fetchBacktest = useCallback(() => {
    setBacktestLoading(true);
    getShortTermBacktestStats()
      .then((res) => {
        if (res?.data?.setups) {
          setBacktestStats(res.data.setups);
        }
      })
      .catch((err) => console.error('[ShortTerm] Backtest fetch error:', err))
      .finally(() => setBacktestLoading(false));
  }, []);

  // Load based on active tab
  useEffect(() => {
    if (activeTab === 'scanner') {
      fetchScans(false);
    } else if (activeTab === 'journal') {
      fetchJournal();
    } else if (activeTab === 'tax') {
      fetchTaxInfo();
    } else if (activeTab === 'backtest') {
      fetchBacktest();
    }
  }, [activeTab, mode, fetchScans, fetchJournal, fetchTaxInfo, fetchBacktest]);

  // Take this trade handler
  const handleTakeTrade = async (setup) => {
    try {
      const shares = setup?.position_sizing?.one_percent_risk?.suggested_shares || 10;
      const capital = setup?.position_sizing?.one_percent_risk?.capital_required_inr || setup.current_price * shares;
      const riskAmount = setup?.position_sizing?.one_percent_risk?.risk_amount_inr || 10000;

      const payload = {
        symbol: setup.symbol,
        setup_name: setup.setup_name,
        mode: setup.mode,
        direction: setup.direction || 'BUY',
        entry_zone_min: setup.entry_zone?.min,
        entry_zone_max: setup.entry_zone?.max,
        entry_price: setup.entry_zone?.recommended_entry || setup.current_price,
        stop_loss: setup.stop_loss,
        target_1: setup.target_1,
        target_2: setup.target_2,
        target_3: setup.target_3,
        time_stop_days: setup.time_stop?.days || 5,
        quantity: shares,
        capital_allocated: capital,
        risk_amount: riskAmount,
        risk_reward: setup.risk_reward_ratio,
        confidence: setup.confidence_score,
        notes: `Logged via Short-Term Scanner. Setup: ${setup.setup_name}. ${setup.time_stop?.description || ''}`,
      };

      await addToJournal(payload);
      showToast(`✅ Trade for ${setup.symbol} added to Journal!`);
      if (activeTab === 'journal') fetchJournal();
    } catch (err) {
      console.error('Failed to log trade:', err);
      showToast(`❌ Error logging trade: ${err.message}`);
    }
  };

  // Journal trade status updater
  const handleUpdateStatus = async (tradeId, newStatus, exitPrice = null) => {
    try {
      const payload = { status: newStatus };
      if (exitPrice !== null) payload.exit_price = exitPrice;
      await updateJournal(tradeId, payload);
      showToast(`Status updated to ${newStatus}`);
      fetchJournal();
    } catch (err) {
      showToast(`Failed to update status: ${err.message}`);
    }
  };

  const handleDeleteTrade = async (tradeId) => {
    if (!window.confirm('Delete this trade record?')) return;
    try {
      await deleteFromJournal(tradeId);
      showToast('Trade record deleted');
      fetchJournal();
    } catch (err) {
      showToast(`Failed to delete: ${err.message}`);
    }
  };

  return (
    <div className="space-y-6">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 bg-blue-600 text-white px-5 py-3 rounded-xl shadow-2xl flex items-center gap-3 animate-fade-in border border-blue-400">
          <span>⚡</span>
          <p className="text-sm font-semibold">{toastMessage}</p>
        </div>
      )}

      {/* Header Banner */}
      <div className="bg-gradient-to-r from-gray-900 via-blue-950 to-gray-900 border border-blue-800/40 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xl">⚡</span>
              <h1 className="text-2xl font-bold text-white tracking-tight">Short-Term Gain Engine</h1>
              <span className="bg-blue-500/20 text-blue-400 text-xs px-2.5 py-0.5 rounded-full border border-blue-500/30 font-semibold">
                NSE High-Probability
              </span>
            </div>
            <p className="text-sm text-gray-300 max-w-2xl">
              High-velocity breakout, pullback, and momentum setups with institutional 1:2 Risk-Reward enforcement,
              adaptive ATR stops, time-stop rules, and post-Budget 2024 Indian capital gains tax tracking.
            </p>
          </div>

          {/* Mode Switcher */}
          <div className="bg-gray-800/90 border border-gray-700 p-1.5 rounded-xl flex items-center gap-1 shadow-inner self-start md:self-auto">
            <button
              onClick={() => setMode('INTRADAY')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                mode === 'INTRADAY'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
              }`}
            >
              ⚡ Intraday (1D)
            </button>
            <button
              onClick={() => setMode('SWING')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                mode === 'SWING'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
              }`}
            >
              🎯 Swing (2-5D)
            </button>
            <button
              onClick={() => setMode('SHORT_TERM')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                mode === 'SHORT_TERM'
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700/50'
              }`}
            >
              🚀 Short-Term (1-3W)
            </button>
          </div>
        </div>

        {/* Mode Info Pill */}
        <div className="mt-4 pt-3 border-t border-gray-800 flex flex-wrap items-center gap-4 text-xs text-gray-400">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
            Mode: <strong className="text-gray-200">{mode}</strong>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-yellow-400"></span>
            Stop-Loss Rule:{' '}
            <strong className="text-gray-200">
              {mode === 'INTRADAY' ? '1.0x ATR (Intraday Tight)' : mode === 'SWING' ? '1.5x ATR (Swing)' : '2.0x ATR (Multi-Week)'}
            </strong>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400"></span>
            Mandatory R:R: <strong className="text-green-400">Minimum 1:2 (Enforced)</strong>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            Time Stop:{' '}
            <strong className="text-gray-200">
              {mode === 'INTRADAY' ? '3:15 PM EOD' : mode === 'SWING' ? '5 Trading Days' : '15 Trading Days'}
            </strong>
          </span>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex border-b border-gray-800 gap-2">
        <button
          onClick={() => setActiveTab('scanner')}
          className={`pb-3 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
            activeTab === 'scanner'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <span>⚡</span>
          High-Probability Scanner
          {scans.length > 0 && (
            <span className="bg-blue-600/30 text-blue-300 text-xs px-2 py-0.5 rounded-full border border-blue-500/30">
              {scans.length}
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab('journal')}
          className={`pb-3 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
            activeTab === 'journal'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <span>📓</span>
          Trade Journal & P&L
          {journalData?.trades?.length > 0 && (
            <span className="bg-purple-600/30 text-purple-300 text-xs px-2 py-0.5 rounded-full border border-purple-500/30">
              {journalData.trades.length}
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab('tax')}
          className={`pb-3 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
            activeTab === 'tax'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <span>💰</span>
          Indian STCG vs LTCG Tax
        </button>

        <button
          onClick={() => setActiveTab('backtest')}
          className={`pb-3 px-4 text-sm font-semibold border-b-2 transition-all flex items-center gap-2 ${
            activeTab === 'backtest'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <span>📊</span>
          3-5Y Setup Backtests
        </button>
      </div>

      {/* ================= TAB 1: SCANNER ================= */}
      {activeTab === 'scanner' && (
        <ErrorBoundary name="ShortTermScanner" title="Scanner Error">
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap items-center gap-3">
                {/* Setup Selector */}
                <div>
                  <label className="text-[11px] font-semibold text-gray-400 block mb-1">SETUP TYPE</label>
                  <select
                    value={selectedSetup}
                    onChange={(e) => setSelectedSetup(e.target.value)}
                    className="bg-gray-800 border border-gray-700 text-xs text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
                  >
                    {SETUPS.map((s) => (
                      <option key={s.code} value={s.code}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Sector Selector */}
                <div>
                  <label className="text-[11px] font-semibold text-gray-400 block mb-1">SECTOR</label>
                  <select
                    value={selectedSector}
                    onChange={(e) => setSelectedSector(e.target.value)}
                    className="bg-gray-800 border border-gray-700 text-xs text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500"
                  >
                    {SECTORS.map((sec) => (
                      <option key={sec} value={sec}>
                        {sec}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Confidence Slider */}
                <div>
                  <label className="text-[11px] font-semibold text-gray-400 block mb-1">
                    MIN CONFIDENCE: <span className="text-blue-400 font-bold">{minConfidence}%</span>
                  </label>
                  <input
                    type="range"
                    min="40"
                    max="90"
                    step="5"
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(Number(e.target.value))}
                    className="w-28 accent-blue-500"
                  />
                </div>
              </div>

              {/* Refresh & Last updated */}
              <div className="flex items-center gap-3">
                {lastRefreshed && (
                  <span className="text-xs text-gray-500">
                    Refreshed: <span className="text-gray-300">{lastRefreshed}</span>
                  </span>
                )}
                <button
                  onClick={() => fetchScans(true)}
                  disabled={scannerLoading}
                  className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                >
                  <span className={scannerLoading ? 'animate-spin' : ''}>🔄</span>
                  {scannerLoading ? 'Scanning Universe...' : 'Rescan Now'}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {scannerError && (
              <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-4 text-xs text-red-300">
                ⚠️ {scannerError}
              </div>
            )}

            {/* Loading Skeleton */}
            {scannerLoading && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {[1, 2, 3, 4, 5, 6].map((idx) => (
                  <div key={idx} className="bg-gray-900 border border-gray-800 rounded-xl p-5 animate-pulse space-y-4">
                    <div className="h-6 bg-gray-800 rounded w-1/2"></div>
                    <div className="h-4 bg-gray-800 rounded w-3/4"></div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="h-10 bg-gray-800 rounded"></div>
                      <div className="h-10 bg-gray-800 rounded"></div>
                    </div>
                    <div className="h-8 bg-gray-800 rounded"></div>
                  </div>
                ))}
              </div>
            )}

            {/* Results Grid */}
            {!scannerLoading && scans.length === 0 && (
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-12 text-center space-y-3">
                <span className="text-4xl">🔍</span>
                <h3 className="text-lg font-bold text-gray-300">No Setups Matched Filters</h3>
                <p className="text-xs text-gray-500 max-w-md mx-auto">
                  No stocks in the liquid universe currently meet all strict criteria for {selectedSetup} with minimum{' '}
                  {minConfidence}% confidence. Try switching to "All Setups" or lower the confidence filter.
                </p>
                <button
                  onClick={() => {
                    setSelectedSetup('ALL');
                    setSelectedSector('All');
                    setMinConfidence(45);
                  }}
                  className="mt-2 bg-gray-800 hover:bg-gray-700 text-blue-400 text-xs px-4 py-2 rounded-lg border border-gray-700 font-semibold"
                >
                  Reset Filters
                </button>
              </div>
            )}

            {!scannerLoading && scans.length > 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {scans.map((setup) => (
                  <div
                    key={setup.symbol}
                    className="bg-gray-900 border border-gray-800 hover:border-blue-600/50 rounded-2xl p-5 shadow-lg flex flex-col justify-between transition-all"
                  >
                    <div>
                      {/* Top row */}
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div>
                          <div className="flex items-center gap-2">
                            <h3 className="text-lg font-bold text-white tracking-wide">{setup.symbol}</h3>
                            <span className="text-[10px] bg-gray-800 text-gray-400 px-2 py-0.5 rounded font-medium border border-gray-700">
                              {setup.sector}
                            </span>
                          </div>
                          <p className="text-xs text-gray-400 truncate max-w-[200px]">{setup.company_name}</p>
                        </div>
                        <div className="text-right">
                          <div className="text-base font-extrabold text-white">₹{setup.current_price?.toLocaleString('en-IN')}</div>
                          <span
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                              setup.status === 'ACTIONABLE'
                                ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/60'
                                : 'bg-amber-950/80 text-amber-400 border border-amber-800/60'
                            }`}
                          >
                            {setup.status === 'ACTIONABLE' ? '⚡ ACTIONABLE' : '⚠️ NO TRADE'}
                          </span>
                        </div>
                      </div>

                      {/* Setup Name & Badge */}
                      <div className="bg-blue-950/40 border border-blue-900/50 rounded-lg p-2.5 my-3 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-base">📌</span>
                          <div>
                            <p className="text-xs font-bold text-blue-300">{setup.setup_name}</p>
                            <p className="text-[10px] text-gray-400">{setup.entry_zone?.trigger_condition}</p>
                          </div>
                        </div>
                      </div>

                      {/* Metrics row */}
                      <div className="grid grid-cols-3 gap-2 my-3 text-center">
                        <div className="bg-gray-800/60 p-2 rounded-lg border border-gray-800">
                          <span className="text-[10px] text-gray-400 block">CONFIDENCE</span>
                          <span className="text-xs font-bold text-blue-400">{setup.confidence_score}%</span>
                        </div>
                        <div className="bg-gray-800/60 p-2 rounded-lg border border-gray-800">
                          <span className="text-[10px] text-gray-400 block">5Y WIN RATE</span>
                          <span className="text-xs font-bold text-emerald-400">{setup.historical_win_rate_pct}%</span>
                        </div>
                        <div className="bg-gray-800/60 p-2 rounded-lg border border-gray-800">
                          <span className="text-[10px] text-gray-400 block">RISK:REWARD</span>
                          <span className="text-xs font-bold text-cyan-400">1:{setup.risk_reward_ratio}</span>
                        </div>
                      </div>

                      {/* Trade Levels */}
                      <div className="space-y-1.5 text-xs bg-gray-950/60 p-3 rounded-xl border border-gray-800/80 my-3">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-400 font-medium">Entry Zone:</span>
                          <span className="font-bold text-white">
                            ₹{setup.entry_zone?.min} - ₹{setup.entry_zone?.max}
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-red-400">
                          <span>Stop-Loss:</span>
                          <span className="font-bold">
                            ₹{setup.stop_loss} ({setup.sl_distance_pct}% risk)
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-emerald-400">
                          <span>Target 1 (1:2 R:R):</span>
                          <span className="font-bold">₹{setup.target_1} (+{setup.targets?.[0]?.gain_pct}%)</span>
                        </div>
                        {setup.target_2 && (
                          <div className="flex justify-between items-center text-emerald-300">
                            <span>Target 2 (1:3 R:R):</span>
                            <span className="font-bold">₹{setup.target_2} (+{setup.targets?.[1]?.gain_pct}%)</span>
                          </div>
                        )}
                        <div className="pt-1 border-t border-gray-800 flex justify-between items-center text-[11px] text-purple-400">
                          <span>Time Stop:</span>
                          <span>{setup.time_stop?.days} days ({setup.time_stop?.description})</span>
                        </div>
                      </div>

                      {/* Narrative */}
                      <div className="text-[11px] space-y-1 text-gray-400 my-2">
                        <p>
                          <strong className="text-gray-300">Why this setup:</strong> {setup.analysis_narrative?.why_this_setup}
                        </p>
                        <p>
                          <strong className="text-rose-400">Invalidation:</strong>{' '}
                          {setup.analysis_narrative?.what_invalidates_it}
                        </p>
                      </div>

                      {/* Position sizing snippet */}
                      <div className="bg-gray-800/40 p-2.5 rounded-lg border border-gray-800 text-[11px] text-gray-300 space-y-1 mb-3">
                        <div className="flex justify-between text-gray-400">
                          <span>Position (1% Capital Risk):</span>
                          <strong className="text-white">
                            {setup.position_sizing?.one_percent_risk?.suggested_shares} shares (~₹
                            {setup.position_sizing?.one_percent_risk?.capital_required_inr?.toLocaleString('en-IN')})
                          </strong>
                        </div>
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div className="pt-2 border-t border-gray-800 flex gap-2">
                      <button
                        onClick={() => handleTakeTrade(setup)}
                        className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-2 px-3 rounded-xl text-xs flex items-center justify-center gap-1.5 shadow-md transition-all active:scale-95"
                      >
                        <span>⚡</span> Take This Trade
                      </button>
                      <a
                        href={`/?symbol=${setup.symbol}`}
                        className="bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold py-2 px-3 rounded-xl text-xs flex items-center justify-center transition-colors border border-gray-700"
                        title="View Interactive TradingView Chart"
                      >
                        📈 Chart
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ErrorBoundary>
      )}

      {/* ================= TAB 2: TRADE JOURNAL ================= */}
      {activeTab === 'journal' && (
        <ErrorBoundary name="TradeJournal" title="Journal Error">
          <div className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <span className="text-xs text-gray-400 block font-medium">TOTAL TRADES</span>
                <span className="text-xl font-bold text-white">{journalData?.summary?.total_trades || 0}</span>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <span className="text-xs text-gray-400 block font-medium">ACTIVE POSITIONS</span>
                <span className="text-xl font-bold text-blue-400">{journalData?.summary?.open_trades || 0}</span>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <span className="text-xs text-gray-400 block font-medium">CLOSED TRADES</span>
                <span className="text-xl font-bold text-purple-400">{journalData?.summary?.closed_trades || 0}</span>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <span className="text-xs text-gray-400 block font-medium">WIN RATE</span>
                <span className="text-xl font-bold text-emerald-400">{journalData?.summary?.win_rate_pct || 0}%</span>
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-center">
                <span className="text-xs text-gray-400 block font-medium">REALIZED P&L</span>
                <span
                  className={`text-xl font-bold ${
                    (journalData?.summary?.total_realized_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  ₹{(journalData?.summary?.total_realized_pnl || 0).toLocaleString('en-IN')}
                </span>
              </div>
            </div>

            {/* Trades Table */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-xl">
              <div className="px-5 py-4 border-b border-gray-800 flex justify-between items-center">
                <div>
                  <h3 className="text-sm font-bold text-white">Execution Log & Journal</h3>
                  <p className="text-xs text-gray-400">Track entries, stop-loss adjustments, target fills, and net P&L</p>
                </div>
                <button
                  onClick={fetchJournal}
                  className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded-lg border border-gray-700 font-semibold"
                >
                  🔄 Refresh
                </button>
              </div>

              {journalLoading && (
                <div className="p-8 text-center text-xs text-gray-400 animate-pulse">Loading journal trades...</div>
              )}

              {!journalLoading && (!journalData.trades || journalData.trades.length === 0) && (
                <div className="p-12 text-center space-y-2">
                  <span className="text-3xl">📓</span>
                  <h4 className="text-sm font-bold text-gray-300">No Trades in Journal Yet</h4>
                  <p className="text-xs text-gray-500">
                    Click "⚡ Take This Trade" on any setup card in the Scanner to log your first trade plan.
                  </p>
                </div>
              )}

              {!journalLoading && journalData.trades?.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-gray-950/70 text-gray-400 uppercase text-[10px] tracking-wider border-b border-gray-800">
                      <tr>
                        <th className="py-3 px-4">Stock</th>
                        <th className="py-3 px-4">Setup / Mode</th>
                        <th className="py-3 px-4">Entry</th>
                        <th className="py-3 px-4">Stop-Loss</th>
                        <th className="py-3 px-4">Targets</th>
                        <th className="py-3 px-4">Qty</th>
                        <th className="py-3 px-4">Status</th>
                        <th className="py-3 px-4">Realized P&L</th>
                        <th className="py-3 px-4 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800/60">
                      {journalData.trades.map((trade) => (
                        <tr key={trade.id} className="hover:bg-gray-800/30 transition-colors">
                          <td className="py-3 px-4">
                            <span className="font-bold text-white block">{trade.symbol}</span>
                            <span className="text-[10px] text-gray-500">{trade.entry_date}</span>
                          </td>
                          <td className="py-3 px-4">
                            <span className="font-medium text-gray-300 block">{trade.setup_name}</span>
                            <span className="text-[10px] text-blue-400">{trade.mode}</span>
                          </td>
                          <td className="py-3 px-4 font-semibold text-white">₹{trade.entry_price}</td>
                          <td className="py-3 px-4 font-semibold text-rose-400">₹{trade.stop_loss}</td>
                          <td className="py-3 px-4 text-emerald-400">
                            <div>T1: ₹{trade.target_1}</div>
                            {trade.target_2 && <div className="text-[10px] text-emerald-300">T2: ₹{trade.target_2}</div>}
                          </td>
                          <td className="py-3 px-4 font-medium text-gray-300">{trade.quantity}</td>
                          <td className="py-3 px-4">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                trade.status === 'OPEN'
                                  ? 'bg-blue-950 text-blue-400 border border-blue-800'
                                  : trade.status?.includes('HIT')
                                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                                  : 'bg-gray-800 text-gray-400'
                              }`}
                            >
                              {trade.status}
                            </span>
                          </td>
                          <td className="py-3 px-4">
                            {trade.pnl !== null ? (
                              <span
                                className={`font-bold ${
                                  trade.pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                                }`}
                              >
                                ₹{trade.pnl} ({trade.pnl_pct}%)
                              </span>
                            ) : (
                              <span className="text-gray-500">—</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-right">
                            <div className="flex items-center justify-end gap-1.5">
                              {trade.status === 'OPEN' && (
                                <>
                                  <button
                                    onClick={() => handleUpdateStatus(trade.id, 'T1_HIT', trade.target_1)}
                                    className="bg-emerald-800/40 hover:bg-emerald-700/60 text-emerald-300 px-2 py-1 rounded text-[10px] font-semibold border border-emerald-700"
                                    title="Mark Target 1 Hit"
                                  >
                                    T1 Hit
                                  </button>
                                  <button
                                    onClick={() => handleUpdateStatus(trade.id, 'SL_HIT', trade.stop_loss)}
                                    className="bg-rose-900/40 hover:bg-rose-800/60 text-rose-300 px-2 py-1 rounded text-[10px] font-semibold border border-rose-800"
                                    title="Mark Stop Loss Hit"
                                  >
                                    SL Hit
                                  </button>
                                </>
                              )}
                              <button
                                onClick={() => handleDeleteTrade(trade.id)}
                                className="bg-gray-800 hover:bg-red-900/50 text-gray-400 hover:text-red-300 px-2 py-1 rounded text-[10px] transition-colors"
                                title="Delete Record"
                              >
                                ✕
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </ErrorBoundary>
      )}

      {/* ================= TAB 3: CAPITAL GAINS TAX ================= */}
      {activeTab === 'tax' && (
        <ErrorBoundary name="CapitalGainsTax" title="Tax Calculator Error">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Form */}
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 space-y-4 shadow-xl">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span>💰</span> Trade Details
                </h3>
                <p className="text-xs text-gray-400">Post-Union Budget 2024 statutory tax computation</p>
              </div>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="text-gray-300 font-semibold block mb-1">Buy Price (₹)</label>
                  <input
                    type="number"
                    value={taxInputs.buy_price}
                    onChange={(e) => setTaxInputs({ ...taxInputs, buy_price: Number(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white font-medium focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="text-gray-300 font-semibold block mb-1">Sell Price (₹)</label>
                  <input
                    type="number"
                    value={taxInputs.sell_price}
                    onChange={(e) => setTaxInputs({ ...taxInputs, sell_price: Number(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white font-medium focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="text-gray-300 font-semibold block mb-1">Quantity (Shares)</label>
                  <input
                    type="number"
                    value={taxInputs.quantity}
                    onChange={(e) => setTaxInputs({ ...taxInputs, quantity: Number(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white font-medium focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-gray-300 font-semibold block mb-1">Buy Date</label>
                    <input
                      type="date"
                      value={taxInputs.buy_date}
                      onChange={(e) => setTaxInputs({ ...taxInputs, buy_date: e.target.value })}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-white text-xs focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="text-gray-300 font-semibold block mb-1">Sell Date</label>
                    <input
                      type="date"
                      value={taxInputs.sell_date}
                      onChange={(e) => setTaxInputs({ ...taxInputs, sell_date: e.target.value })}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-white text-xs focus:outline-none focus:border-blue-500"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-gray-300 font-semibold block mb-1">Trade Category</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setTaxInputs({ ...taxInputs, trade_type: 'DELIVERY' })}
                      className={`py-2 text-xs font-semibold rounded-lg border transition-all ${
                        taxInputs.trade_type === 'DELIVERY'
                          ? 'bg-blue-600 text-white border-blue-500'
                          : 'bg-gray-800 text-gray-400 border-gray-700'
                      }`}
                    >
                      Delivery (CNC)
                    </button>
                    <button
                      type="button"
                      onClick={() => setTaxInputs({ ...taxInputs, trade_type: 'INTRADAY' })}
                      className={`py-2 text-xs font-semibold rounded-lg border transition-all ${
                        taxInputs.trade_type === 'INTRADAY'
                          ? 'bg-blue-600 text-white border-blue-500'
                          : 'bg-gray-800 text-gray-400 border-gray-700'
                      }`}
                    >
                      Intraday (MIS)
                    </button>
                  </div>
                </div>

                <div>
                  <label className="text-gray-300 font-semibold block mb-1">Flat Brokerage / Order (₹)</label>
                  <input
                    type="number"
                    value={taxInputs.brokerage_per_order}
                    onChange={(e) => setTaxInputs({ ...taxInputs, brokerage_per_order: Number(e.target.value) })}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white font-medium focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <button
                onClick={fetchTaxInfo}
                disabled={taxLoading}
                className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-2.5 rounded-xl text-xs shadow-lg transition-all"
              >
                {taxLoading ? 'Recalculating...' : '⚡ Compute Post-Tax Profit'}
              </button>
            </div>

            {/* Results Panel */}
            <div className="lg:col-span-2 space-y-4">
              {taxResult && (
                <>
                  {/* Top Summary Banner */}
                  <div className="bg-gradient-to-r from-emerald-950/60 to-blue-950/60 border border-emerald-800/40 rounded-2xl p-5 shadow-xl">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div>
                        <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                          {taxResult.summary?.classification} • {taxResult.tax_breakdown?.tax_section}
                        </span>
                        <h2 className="text-2xl font-extrabold text-white mt-0.5">
                          ₹{taxResult.summary?.net_profit_in_hand?.toLocaleString('en-IN')}
                        </h2>
                        <p className="text-xs text-gray-300">
                          Net Post-Tax In-Pocket Profit (ROI: <strong className="text-emerald-400">+{taxResult.summary?.roi_pct}%</strong>)
                        </p>
                      </div>
                      <div className="text-right text-xs space-y-1">
                        <div className="text-gray-400">
                          Gross Capital Gain: <strong className="text-white">₹{taxResult.summary?.gross_profit?.toLocaleString('en-IN')}</strong>
                        </div>
                        <div className="text-gray-400">
                          Total Friction Charges: <strong className="text-rose-400">₹{taxResult.summary?.total_charges}</strong>
                        </div>
                        <div className="text-gray-400">
                          Total Income Tax + Cess: <strong className="text-amber-400">₹{taxResult.summary?.total_tax_liability}</strong>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 12-Month Holding Tax Comparison */}
                  <div className="bg-gray-900 border border-blue-900/40 rounded-2xl p-5 space-y-3 shadow-lg">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold text-blue-300 uppercase tracking-wider">
                        📈 Holding Horizon Comparison (Post-Budget 2024)
                      </h4>
                      {taxResult.holding_period_comparison?.tax_savings_if_held > 0 && (
                        <span className="bg-emerald-500/20 text-emerald-400 text-xs px-2.5 py-0.5 rounded-full font-bold border border-emerald-500/30">
                          Save ₹{taxResult.holding_period_comparison.tax_savings_if_held?.toLocaleString('en-IN')} by holding &gt;12M
                        </span>
                      )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                      <div className="bg-gray-950/70 p-3.5 rounded-xl border border-gray-800 space-y-1.5">
                        <span className="text-[10px] text-amber-400 font-bold uppercase">If Sold Today (STCG 20%)</span>
                        <div className="text-base font-bold text-white">
                          Tax: ₹{taxResult.holding_period_comparison?.sold_today?.tax_amount?.toLocaleString('en-IN')}
                        </div>
                        <div className="text-gray-400">
                          Net In Hand:{' '}
                          <strong className="text-gray-200">
                            ₹{taxResult.holding_period_comparison?.sold_today?.net_in_hand?.toLocaleString('en-IN')}
                          </strong>
                        </div>
                        <p className="text-[10px] text-gray-500">Held for {taxResult.summary?.holding_days} days</p>
                      </div>

                      <div className="bg-gray-950/70 p-3.5 rounded-xl border border-gray-800 space-y-1.5">
                        <span className="text-[10px] text-emerald-400 font-bold uppercase">If Held &gt;12 Months (LTCG 12.5%)</span>
                        <div className="text-base font-bold text-white">
                          Tax: ₹{taxResult.holding_period_comparison?.held_over_12_months?.tax_amount?.toLocaleString('en-IN')}
                        </div>
                        <div className="text-gray-400">
                          Net In Hand:{' '}
                          <strong className="text-gray-200">
                            ₹{taxResult.holding_period_comparison?.held_over_12_months?.net_in_hand?.toLocaleString('en-IN')}
                          </strong>
                        </div>
                        <p className="text-[10px] text-emerald-400/80">Includes ₹1.25 Lakh annual LTCG exemption</p>
                      </div>
                    </div>
                  </div>

                  {/* Charges Breakdown Table */}
                  <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-lg space-y-3">
                    <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider">
                      Itemized Transaction Friction & Statutory Charges
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 text-xs">
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">Brokerage (Buy + Sell)</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.brokerage}</span>
                      </div>
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">Securities Transaction Tax (STT)</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.stt}</span>
                      </div>
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">Exchange Turnover Fee</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.exchange_turnover_charges}</span>
                      </div>
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">SEBI Turnover Charges</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.sebi_turnover_charges}</span>
                      </div>
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">Stamp Duty (Purchase)</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.stamp_duty}</span>
                      </div>
                      <div className="bg-gray-950/60 p-2.5 rounded-lg border border-gray-800/80">
                        <span className="text-gray-500 block text-[10px]">GST @ 18%</span>
                        <span className="font-bold text-white">₹{taxResult.charges_breakdown?.gst_18_pct}</span>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-gray-800 text-[11px] text-gray-500 italic">
                      ⚠️ <strong>Statutory Note:</strong> {taxResult.advisory_note}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </ErrorBoundary>
      )}

      {/* ================= TAB 4: BACKTEST PERFORMANCE ================= */}
      {activeTab === 'backtest' && (
        <ErrorBoundary name="SetupBacktests" title="Backtests Error">
          <div className="space-y-4">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-xl">
              <h3 className="text-base font-bold text-white">Transparent 5-Year Setup Performance</h3>
              <p className="text-xs text-gray-400">
                Backtested on historical NSE equity universe (2019-2024). Includes ₹20 brokerage, 0.1% STT, and 0.05% slippage.
              </p>
            </div>

            {backtestLoading && (
              <div className="p-12 text-center text-xs text-gray-400 animate-pulse">Loading setup backtest history...</div>
            )}

            {!backtestLoading && backtestStats && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(backtestStats).map(([code, stat]) => (
                  <div key={code} className="bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-lg space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-extrabold text-blue-400 uppercase tracking-wide">{code}</span>
                      <span className="text-[10px] bg-gray-800 text-gray-300 px-2 py-0.5 rounded font-medium border border-gray-700">
                        {stat.timeframe}
                      </span>
                    </div>

                    <h4 className="text-sm font-bold text-white">{stat.setup_name}</h4>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-gray-950/70 p-2.5 rounded-lg border border-gray-800">
                        <span className="text-[10px] text-gray-500 block">WIN RATE</span>
                        <span className="text-base font-bold text-emerald-400">{stat.win_rate_pct}%</span>
                      </div>
                      <div className="bg-gray-950/70 p-2.5 rounded-lg border border-gray-800">
                        <span className="text-[10px] text-gray-500 block">PROFIT FACTOR</span>
                        <span className="text-base font-bold text-white">{stat.profit_factor}</span>
                      </div>
                      <div className="bg-gray-950/70 p-2.5 rounded-lg border border-gray-800">
                        <span className="text-[10px] text-gray-500 block">MAX DRAWDOWN</span>
                        <span className="text-base font-bold text-rose-400">-{stat.max_drawdown_pct}%</span>
                      </div>
                      <div className="bg-gray-950/70 p-2.5 rounded-lg border border-gray-800">
                        <span className="text-[10px] text-gray-500 block">AVG HOLDING</span>
                        <span className="text-base font-bold text-purple-400">{stat.avg_holding_days} Days</span>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-gray-800 text-[10px] text-gray-400 space-y-0.5">
                      <div>Total Sample Trades: <strong className="text-gray-200">{stat.total_trades}</strong></div>
                      <div>Sharpe Ratio: <strong className="text-gray-200">{stat.sharpe_ratio}</strong></div>
                      <div>Period: <strong className="text-gray-200">{stat.backtest_period}</strong></div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </ErrorBoundary>
      )}
    </div>
  );
}
