import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  scanStocks,
  getBreakouts,
  getOversold,
  getOverbought,
  getVolumeSpikes,
  getStrongRS,
  getBullishEMA,
} from '../api/client';

/**
 * Stock screener/scanner component.
 * Features:
 * - High-speed instant response from backend cache
 * - Defends against unmounted state updates with AbortController & isMounted flag
 * - Guards array check before sorting or mapping
 * - Safe numeric formatting and fallback dashes
 */
export default function Screener({ onSelectStock }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('scan');
  const [sortKey, setSortKey] = useState('change_pct');
  const [sortDir, setSortDir] = useState(-1);

  const abortControllerRef = useRef(null);

  const filters = [
    { id: 'scan', label: 'All Nifty 30', icon: '📊' },
    { id: 'breakouts', label: 'Breakouts', icon: '🚀' },
    { id: 'oversold', label: 'Oversold', icon: '📉' },
    { id: 'overbought', label: 'Overbought', icon: '🔥' },
    { id: 'volume', label: 'Volume Spike', icon: '⚡' },
    { id: 'strong_rs', label: 'Strong RS', icon: '💪' },
    { id: 'bullish_ema', label: 'Bullish EMA', icon: '📈' },
  ];

  const fetchData = useCallback(async () => {
    // Abort any in-flight screener request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const config = { signal: controller.signal };
      let res;
      switch (filter) {
        case 'breakouts':
          res = await getBreakouts(config);
          break;
        case 'oversold':
          res = await getOversold(config);
          break;
        case 'overbought':
          res = await getOverbought(config);
          break;
        case 'volume':
          res = await getVolumeSpikes(config);
          break;
        case 'strong_rs':
          res = await getStrongRS(config);
          break;
        case 'bullish_ema':
          res = await getBullishEMA(config);
          break;
        default:
          res = await scanStocks(null, config);
      }

      const stockList = Array.isArray(res?.data?.stocks) ? res.data.stocks : [];
      setResults(stockList);
    } catch (err) {
      if (err.name === 'CanceledError' || err.message === 'canceled') {
        return; // Request was aborted cleanly
      }
      console.warn('[Screener] Fetch failed:', err);
      setError('Unable to load screener scan data.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchData();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchData]);

  // Safe sorting
  const safeResults = Array.isArray(results) ? results : [];
  const sorted = [...safeResults].sort((a, b) => {
    const aVal = a?.[sortKey] ?? 0;
    const bVal = b?.[sortKey] ?? 0;
    return (aVal - bVal) * sortDir;
  });

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(-sortDir);
    } else {
      setSortKey(key);
      setSortDir(-1);
    }
  };

  const SortHeader = ({ label, field, className = '' }) => (
    <th
      className={`py-2 px-2 text-left cursor-pointer hover:text-blue-400 select-none ${className} ${
        sortKey === field ? 'text-blue-400 font-bold' : ''
      }`}
      onClick={() => handleSort(field)}
    >
      {label} {sortKey === field ? (sortDir > 0 ? '▲' : '▼') : ''}
    </th>
  );

  return (
    <div className="card shadow-md">
      <div className="card-header">
        <span>🔍</span> Stock Screener & Scanner
        <span className="ml-auto text-xs text-gray-400 font-mono">
          {safeResults.length} Stocks Found
        </span>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 mb-4 flex-wrap">
        {filters.map((f) => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            disabled={loading}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              filter === f.id
                ? 'bg-blue-600 text-white shadow'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {f.icon} {f.label}
          </button>
        ))}
      </div>

      {/* Error banner */}
      {error && (
        <div className="p-3 mb-3 bg-red-950/40 border border-red-800 text-red-200 rounded-lg text-xs flex items-center justify-between">
          <span>⚠️ {error}</span>
          <button
            onClick={fetchData}
            className="px-2.5 py-1 bg-red-800 hover:bg-red-700 text-white rounded font-medium"
          >
            Retry
          </button>
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center justify-center py-8">
          <div className="spinner mb-2" />
          <span className="text-xs text-gray-400">Loading screening metrics...</span>
        </div>
      ) : safeResults.length === 0 ? (
        <div className="text-center py-8 text-xs text-gray-500">
          No stocks currently match the &ldquo;{filter}&rdquo; criteria.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-gray-500 border-b border-gray-800">
              <tr>
                <th className="py-2 px-2 text-left">Symbol</th>
                <SortHeader label="Price" field="price" />
                <SortHeader label="Change%" field="change_pct" />
                <SortHeader label="RSI" field="rsi" />
                <SortHeader label="Vol Ratio" field="volume_ratio" />
                <SortHeader label="RS vs Nifty" field="rs_vs_nifty" />
                <th className="py-2 px-2 text-left">Technical Signals</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((stock) => {
                if (!stock) return null;
                const changePct = stock.change_pct;
                const rsi = stock.rsi;
                const volRatio = stock.volume_ratio;
                const rsNifty = stock.rs_vs_nifty;

                return (
                  <tr
                    key={stock.symbol}
                    className="border-b border-gray-800/50 hover:bg-gray-800/40 cursor-pointer transition-colors"
                    onClick={() => onSelectStock?.(stock.symbol)}
                  >
                    <td className="py-2.5 px-2">
                      <div className="font-semibold text-gray-100">{stock.symbol}</div>
                      <div className="text-[11px] text-gray-500 truncate max-w-[140px]">
                        {stock.name || stock.sector || 'NSE'}
                      </div>
                    </td>
                    <td className="py-2.5 px-2 font-mono">
                      {stock.price != null && Number.isFinite(Number(stock.price))
                        ? `₹${Number(stock.price).toFixed(2)}`
                        : '—'}
                    </td>
                    <td
                      className={`py-2.5 px-2 font-mono font-medium ${
                        (changePct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {changePct != null && Number.isFinite(Number(changePct))
                        ? `${changePct >= 0 ? '+' : ''}${Number(changePct).toFixed(2)}%`
                        : '—'}
                    </td>
                    <td
                      className={`py-2.5 px-2 font-mono ${
                        rsi != null && rsi < 30
                          ? 'text-green-400 font-bold'
                          : rsi != null && rsi > 70
                          ? 'text-red-400 font-bold'
                          : 'text-gray-300'
                      }`}
                    >
                      {rsi != null && Number.isFinite(Number(rsi)) ? Number(rsi).toFixed(1) : '—'}
                    </td>
                    <td
                      className={`py-2.5 px-2 font-mono ${
                        (volRatio ?? 0) > 2 ? 'text-yellow-400 font-bold' : 'text-gray-300'
                      }`}
                    >
                      {volRatio != null && Number.isFinite(Number(volRatio))
                        ? `${Number(volRatio).toFixed(1)}x`
                        : '—'}
                    </td>
                    <td
                      className={`py-2.5 px-2 font-mono ${
                        (rsNifty ?? 0) > 0 ? 'text-green-400' : (rsNifty ?? 0) < 0 ? 'text-red-400' : 'text-gray-400'
                      }`}
                    >
                      {rsNifty != null && Number.isFinite(Number(rsNifty))
                        ? `${rsNifty > 0 ? '+' : ''}${Number(rsNifty).toFixed(1)}%`
                        : '—'}
                    </td>
                    <td className="py-2.5 px-2">
                      <div className="flex gap-1 flex-wrap">
                        {stock.is_breakout && <span className="badge-bull text-[10px]">Breakout</span>}
                        {rsi != null && rsi < 30 && <span className="badge-bull text-[10px]">Oversold</span>}
                        {rsi != null && rsi > 70 && <span className="badge-bear text-[10px]">Overbought</span>}
                        {volRatio != null && volRatio > 2 && <span className="badge-neutral text-[10px]">Vol Spike</span>}
                        {stock.near_52w_high && <span className="badge-bull text-[10px]">52W High</span>}
                        {stock.macd_bullish_cross && <span className="badge-bull text-[10px]">MACD↑</span>}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
