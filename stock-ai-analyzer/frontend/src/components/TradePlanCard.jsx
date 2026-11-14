import React from 'react';

/**
 * Trade plan card showing entry, SL, targets, R:R, position sizing, and summary.
 * Safely guards against undefined, null, or missing fields.
 */
export default function TradePlanCard({ plan }) {
  if (!plan) return null;

  const direction = plan.direction || 'NEUTRAL';
  const isNoTrade = direction === 'NO_TRADE';
  const isBuy = direction === 'BUY';

  // Support both explicit targets array or discrete target_1/2/3 fields
  const targets = Array.isArray(plan.targets)
    ? plan.targets
    : [plan.target_1, plan.target_2, plan.target_3].filter((t) => t != null && !Number.isNaN(Number(t)));

  const entry = Number(plan.entry_price);
  const sl = Number(plan.stop_loss);
  const t1 = targets[0] != null ? Number(targets[0]) : Number(plan.target_1);
  const t2 = targets[1] != null ? Number(targets[1]) : Number(plan.target_2);
  const t3 = targets[2] != null ? Number(targets[2]) : Number(plan.target_3);

  const RiskRewardBar = () => {
    if (!Number.isFinite(entry) || !Number.isFinite(sl)) return null;
    const risk = Math.abs(entry - sl);
    const maxTarget = Number.isFinite(t3) ? t3 : Number.isFinite(t2) ? t2 : Number.isFinite(t1) ? t1 : entry + risk * 2;
    const range = maxTarget - sl;
    if (range <= 0) return null;

    const entryPct = Math.min(100, Math.max(0, ((entry - sl) / range) * 100));
    const t1Pct = Number.isFinite(t1) ? Math.min(100, Math.max(0, ((t1 - sl) / range) * 100)) : null;
    const t2Pct = Number.isFinite(t2) ? Math.min(100, Math.max(0, ((t2 - sl) / range) * 100)) : null;
    const t3Pct = Number.isFinite(t3) ? Math.min(100, Math.max(0, ((t3 - sl) / range) * 100)) : null;

    return (
      <div className="relative h-6 bg-gray-700 rounded-full overflow-hidden my-3">
        {/* Risk zone (SL to Entry) */}
        <div
          className="absolute top-0 left-0 h-full bg-red-600/60"
          style={{ width: `${entryPct}%` }}
        />
        {/* Reward zone (Entry to last target) */}
        <div
          className="absolute top-0 h-full bg-green-600/60"
          style={{ left: `${entryPct}%`, width: `${Math.max(0, 100 - entryPct)}%` }}
        />
        {/* Entry marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-blue-400"
          style={{ left: `${entryPct}%` }}
        />
        {/* Target markers */}
        {t1Pct != null && (
          <div className="absolute top-0 h-full w-0.5 bg-green-400" style={{ left: `${t1Pct}%` }} />
        )}
        {t2Pct != null && (
          <div className="absolute top-0 h-full w-0.5 bg-green-400" style={{ left: `${t2Pct}%` }} />
        )}
        {t3Pct != null && (
          <div className="absolute top-0 h-full w-0.5 bg-green-300" style={{ left: `${t3Pct}%` }} />
        )}
        {/* Labels */}
        <div className="absolute inset-0 flex items-center justify-between px-2 text-[10px] font-bold">
          <span className="text-red-300">SL</span>
          <span className="text-blue-300">Entry</span>
          <span className="text-green-300">Targets →</span>
        </div>
      </div>
    );
  };

  return (
    <div className={`card border-l-4 ${isNoTrade ? 'border-l-yellow-500' : isBuy ? 'border-l-green-500' : 'border-l-red-500'}`}>
      <div className="card-header">
        <span>📝</span> Trade Plan
        <span className={isNoTrade ? 'badge-neutral' : isBuy ? 'badge-bull' : 'badge-bear'}>
          {direction}
        </span>
        {plan.trade_type && (
          <span className="ml-auto text-xs px-2 py-1 bg-gray-800 rounded text-gray-400">
            {plan.trade_type}
          </span>
        )}
      </div>

      {isNoTrade ? (
        <div className="p-4 bg-yellow-900/20 rounded-lg border border-yellow-800">
          <p className="text-yellow-300 text-sm font-medium mb-2">⚠️ No Trade Recommended</p>
          <p className="text-sm text-gray-300">
            {plan.reason || 'Setup does not meet minimum risk-reward criteria (1:2) or conflicting signals.'}
          </p>
        </div>
      ) : (
        <>
          {/* Price levels */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div className="p-2 bg-blue-900/30 rounded-lg text-center">
              <div className="stat-label">Entry Zone</div>
              <div className="stat-value text-blue-400 text-lg">
                {Number.isFinite(entry) ? `₹${entry.toFixed(2)}` : '—'}
              </div>
            </div>
            <div className="p-2 bg-red-900/30 rounded-lg text-center">
              <div className="stat-label">Stop Loss</div>
              <div className="stat-value text-red-400 text-lg">
                {Number.isFinite(sl) ? `₹${sl.toFixed(2)}` : '—'}
              </div>
            </div>
            <div className="p-2 bg-green-900/30 rounded-lg text-center">
              <div className="stat-label">Target 1</div>
              <div className="stat-value text-green-400 text-lg">
                {Number.isFinite(t1) ? `₹${t1.toFixed(2)}` : '—'}
              </div>
            </div>
            <div className="p-2 bg-green-900/20 rounded-lg text-center">
              <div className="stat-label">Target 2</div>
              <div className="stat-value text-green-300 text-lg">
                {Number.isFinite(t2) ? `₹${t2.toFixed(2)}` : '—'}
              </div>
            </div>
          </div>

          {/* R:R Bar */}
          <RiskRewardBar />

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-3 mb-3">
            <div className="text-center p-2 bg-gray-800/50 rounded">
              <div className="stat-label">Risk:Reward</div>
              <div className="text-lg font-bold text-gray-100">
                1:{plan.risk_reward != null && Number.isFinite(Number(plan.risk_reward)) ? Number(plan.risk_reward).toFixed(1) : '—'}
              </div>
            </div>
            <div className="text-center p-2 bg-gray-800/50 rounded">
              <div className="stat-label">Confidence</div>
              <div
                className={`text-lg font-bold ${
                  (plan.confidence || 0) >= 60
                    ? 'text-green-400'
                    : (plan.confidence || 0) >= 40
                    ? 'text-yellow-400'
                    : 'text-red-400'
                }`}
              >
                {plan.confidence != null ? `${Number(plan.confidence).toFixed(0)}%` : '—'}
              </div>
            </div>
            <div className="text-center p-2 bg-gray-800/50 rounded">
              <div className="stat-label">Position Size</div>
              <div className="text-lg font-bold text-gray-100">
                {plan.position_size != null ? `${plan.position_size} shares` : '—'}
              </div>
            </div>
          </div>

          {/* Partial booking guidance */}
          {Number.isFinite(t1) && (
            <div className="text-xs text-gray-400 mb-3">
              📌 Book 50% profits at T1 (₹{t1.toFixed(2)}), move SL to breakeven, and trail for remaining.
            </div>
          )}
        </>
      )}

      {/* Summary */}
      {plan.summary && (
        <div className="mt-3 p-3 bg-gray-800/50 rounded-lg">
          <p className="text-sm text-gray-300 leading-relaxed">{plan.summary}</p>
        </div>
      )}
    </div>
  );
}
