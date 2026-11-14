import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

/**
 * Backtest results display with equity curve chart and metrics table.
 */
export default function BacktestResults({ results }) {
  if (!results || results.length === 0) return null;

  const MetricCard = ({ label, value, suffix = '', highlight = false }) => (
    <div className="p-2 bg-gray-800/50 rounded text-center">
      <div className="stat-label">{label}</div>
      <div className={`text-lg font-bold ${highlight
        ? (parseFloat(value) >= 0 ? 'text-green-400' : 'text-red-400')
        : 'text-gray-100'
      }`}>
        {value}{suffix}
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      {(Array.isArray(results) ? results : [results]).map((result, idx) => (
        <div key={idx} className="card">
          <div className="card-header">
            <span>📈</span> {result.strategy}
            <span className="ml-auto text-xs text-gray-400">{result.period}</span>
          </div>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
            <MetricCard label="Total Return" value={result.total_return_pct?.toFixed(1)} suffix="%" highlight />
            <MetricCard label="Win Rate" value={result.win_rate?.toFixed(1)} suffix="%" />
            <MetricCard label="Profit Factor" value={result.profit_factor?.toFixed(2)} />
            <MetricCard label="Sharpe Ratio" value={result.sharpe_ratio?.toFixed(2)} />
            <MetricCard label="Max Drawdown" value={result.max_drawdown?.toFixed(1)} suffix="%" highlight />
            <MetricCard label="Total Trades" value={result.total_trades} />
            <MetricCard label="Avg Win" value={result.avg_win?.toFixed(1)} suffix="%" highlight />
            <MetricCard label="Avg Loss" value={result.avg_loss?.toFixed(1)} suffix="%" highlight />
          </div>

          {/* Equity curve */}
          {result.equity_curve && result.equity_curve.length > 0 && (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={result.equity_curve.map((val, i) => ({ index: i, value: val }))}
                  margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="index" hide />
                  <YAxis
                    stroke="#475569"
                    tickFormatter={(v) => `₹${(v / 100000).toFixed(0)}L`}
                    fontSize={10}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    formatter={(v) => [`₹${v.toLocaleString('en-IN')}`, 'Portfolio']}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#3b82f6"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Cost info */}
          <div className="mt-2 text-xs text-gray-500 text-center">
            Includes brokerage (₹20/trade), STT (0.1% on sell), slippage (0.05%)
          </div>
        </div>
      ))}
    </div>
  );
}
