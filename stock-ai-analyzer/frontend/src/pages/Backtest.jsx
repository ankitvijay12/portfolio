import React, { useState, useEffect } from 'react';
import BacktestResults from '../components/BacktestResults';
import ErrorBoundary from '../components/ErrorBoundary';
import { runBacktest, runAllBacktests, getStrategies } from '../api/client';

export default function Backtest() {
  const [symbol, setSymbol] = useState('RELIANCE');
  const [strategy, setStrategy] = useState('ema_crossover');
  const [capital, setCapital] = useState(1000000);
  const [strategiesList, setStrategiesList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    getStrategies()
      .then((res) => {
        const strats = Array.isArray(res?.data?.strategies) ? res.data.strategies : [];
        setStrategiesList(strats);
      })
      .catch((err) => console.error('Failed to load strategies:', err));
  }, []);

  const handleRunSingle = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await runBacktest({
        symbol: symbol.toUpperCase().trim(),
        strategy,
        capital: Number(capital),
      });
      if (res?.data) {
        setResults([res.data]);
      }
    } catch (err) {
      console.error('Backtest error:', err);
      setError('Backtest execution failed. Please verify parameters.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runAllBacktests(symbol.toUpperCase().trim());
      const resList = Array.isArray(res?.data?.results) ? res.data.results : [];
      setResults(resList);
    } catch (err) {
      console.error('Run all backtest error:', err);
      setError('Batch backtest execution failed.');
    } finally {
      setLoading(false);
    }
  };

  const safeResults = Array.isArray(results) ? results : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>📈</span> Quantitative Strategy Backtester (5–10 Years)
        </h1>
        <p className="text-sm text-gray-400 mt-1">
          Historical backtesting with realistic Indian market friction: ₹20 flat brokerage, 0.1% STT on sell, and 0.05% slippage.
        </p>
      </div>

      {/* Configuration Card */}
      <div className="card shadow-md">
        <form onSubmit={handleRunSingle} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div>
            <label className="text-xs text-gray-400 font-semibold uppercase">Stock Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className="mt-1 w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm font-mono focus:outline-none focus:border-blue-500 text-gray-100"
            />
          </div>

          <div>
            <label className="text-xs text-gray-400 font-semibold uppercase">Strategy Model</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="mt-1 w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm focus:outline-none focus:border-blue-500 text-gray-100"
            >
              {strategiesList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs text-gray-400 font-semibold uppercase">Initial Capital (₹)</label>
            <input
              type="number"
              value={capital}
              onChange={(e) => setCapital(e.target.value)}
              step="50000"
              className="mt-1 w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm font-mono focus:outline-none focus:border-blue-500 text-gray-100"
            />
          </div>

          <div className="flex gap-2">
            <button type="submit" disabled={loading} className="btn-primary flex-1 text-sm">
              {loading ? 'Running...' : 'Run Test'}
            </button>
            <button
              type="button"
              onClick={handleRunAll}
              disabled={loading}
              className="btn-secondary text-sm whitespace-nowrap"
            >
              Compare All
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800 text-red-300 rounded-xl text-sm">
          {error}
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-16 space-y-3">
          <div className="spinner" />
          <p className="text-sm text-gray-400">Computing historical simulations with slippage & taxes...</p>
        </div>
      )}

      {!loading && safeResults.length > 0 && (
        <ErrorBoundary name="Backtest Results">
          <div className="space-y-4">
            <h2 className="text-lg font-bold text-gray-200">Simulation Outcomes</h2>
            <BacktestResults results={safeResults} />
          </div>
        </ErrorBoundary>
      )}
    </div>
  );
}
