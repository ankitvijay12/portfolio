import React from 'react';

/**
 * Displays fundamental analysis metrics in a card with scoring.
 * Defends safely against missing metrics, null numbers, or empty API responses.
 */
export default function FundamentalCard({ data }) {
  if (!data) return null;

  const ScoreCircle = ({ score, label }) => {
    const s = typeof score === 'number' && Number.isFinite(score) ? Math.min(100, Math.max(0, Math.round(score))) : 50;
    const color = s >= 70 ? 'text-green-400' : s >= 40 ? 'text-yellow-400' : 'text-red-400';
    const strokeColor = s >= 70 ? '#4ade80' : s >= 40 ? '#facc15' : '#f87171';
    const radius = 18;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (s / 100) * circumference;

    return (
      <div className="flex flex-col items-center">
        <svg width="48" height="48" className="-rotate-90">
          <circle cx="24" cy="24" r={radius} fill="none" stroke="#334155" strokeWidth="3" />
          <circle
            cx="24"
            cy="24"
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="3"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </svg>
        <span className={`text-base font-bold ${color} -mt-8 font-mono`}>{s}</span>
        <span className="text-[10px] text-gray-400 mt-4 uppercase tracking-wider">{label}</span>
      </div>
    );
  };

  const MetricRow = ({ label, value, good, format = 'number' }) => {
    let displayValue = '—';
    let color = 'text-gray-300';

    if (value != null && Number.isFinite(Number(value))) {
      const num = Number(value);
      if (format === 'percent') {
        displayValue = `${(num * 100).toFixed(1)}%`;
      } else if (format === 'ratio') {
        displayValue = num.toFixed(2);
      } else if (format === 'currency') {
        displayValue = `₹${(num / 10000000).toFixed(1)} Cr`;
      } else {
        displayValue = num.toFixed(2);
      }

      if (good !== undefined) {
        color = good ? 'text-green-400' : 'text-red-400';
      }
    } else if (typeof value === 'string' && value.trim()) {
      displayValue = value;
    }

    return (
      <div className="flex justify-between py-1 border-b border-gray-800/50 last:border-0 text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={`font-mono font-medium ${color}`}>{displayValue}</span>
      </div>
    );
  };

  const metrics = data?.metrics || data || {};
  const scores = data?.category_scores || {};

  return (
    <div className="card shadow-md">
      <div className="card-header">
        <span>📋</span> Fundamental Analysis
      </div>

      {/* Composite score circles */}
      {data?.composite_score != null && (
        <div className="flex items-center gap-4 mb-4 p-3 rounded-lg bg-gray-800/50 justify-around flex-wrap">
          <ScoreCircle score={data.composite_score} label="Overall" />
          {scores.valuation != null && <ScoreCircle score={scores.valuation} label="Valuation" />}
          {scores.profitability != null && <ScoreCircle score={scores.profitability} label="Profit" />}
          {scores.growth != null && <ScoreCircle score={scores.growth} label="Growth" />}
          {scores.health != null && <ScoreCircle score={scores.health} label="Health" />}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Valuation */}
        <div>
          <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">Valuation Multiples</h4>
          <MetricRow label="P/E Ratio" value={metrics?.pe_ratio} format="ratio" />
          <MetricRow label="P/B Ratio" value={metrics?.pb_ratio} format="ratio" />
          <MetricRow label="PEG Ratio" value={metrics?.peg_ratio} format="ratio" />
          <MetricRow label="EV/EBITDA" value={metrics?.ev_ebitda} format="ratio" />
        </div>

        {/* Profitability */}
        <div>
          <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">Profitability & Margins</h4>
          <MetricRow label="ROE" value={metrics?.roe} format="percent" good={metrics?.roe != null ? metrics.roe > 0.15 : undefined} />
          <MetricRow label="ROCE" value={metrics?.roce} format="percent" good={metrics?.roce != null ? metrics.roce > 0.15 : undefined} />
          <MetricRow label="OPM" value={metrics?.operating_margin} format="percent" />
          <MetricRow label="NPM" value={metrics?.profit_margin} format="percent" />
        </div>

        {/* Growth */}
        <div>
          <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">Financial Growth</h4>
          <MetricRow label="Revenue Growth" value={metrics?.revenue_growth} format="percent" good={metrics?.revenue_growth != null ? metrics.revenue_growth > 0 : undefined} />
          <MetricRow label="Earnings Growth" value={metrics?.earnings_growth} format="percent" good={metrics?.earnings_growth != null ? metrics.earnings_growth > 0 : undefined} />
          <MetricRow label="EPS (TTM)" value={metrics?.eps_trailing} format="ratio" />
          <MetricRow label="EPS (Fwd)" value={metrics?.eps_forward} format="ratio" />
        </div>

        {/* Health */}
        <div>
          <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">Balance Sheet Health</h4>
          <MetricRow label="Debt/Equity" value={metrics?.debt_to_equity} format="ratio" good={metrics?.debt_to_equity != null ? metrics.debt_to_equity < 100 : undefined} />
          <MetricRow label="Current Ratio" value={metrics?.current_ratio} format="ratio" good={metrics?.current_ratio != null ? metrics.current_ratio > 1 : undefined} />
          <MetricRow label="Free Cash Flow" value={metrics?.free_cash_flow} format="currency" good={metrics?.free_cash_flow != null ? metrics.free_cash_flow > 0 : undefined} />
          <MetricRow label="Dividend Yield" value={metrics?.dividend_yield} format="percent" />
        </div>
      </div>
    </div>
  );
}
