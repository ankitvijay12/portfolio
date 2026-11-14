import React from 'react';

/**
 * Displays technical indicator values with signal badges and confidence bars.
 * Guards defensively against null/undefined data and nested indicators.
 */
export default function IndicatorPanel({ data }) {
  if (!data) return null;

  const SignalBadge = ({ signal }) => {
    const sig = typeof signal === 'string' ? signal.toUpperCase() : 'NEUTRAL';
    const cls =
      sig === 'BUY' || sig === 'BULLISH'
        ? 'badge-bull'
        : sig === 'SELL' || sig === 'BEARISH'
        ? 'badge-bear'
        : 'badge-neutral';
    return <span className={cls}>{sig}</span>;
  };

  const ConfidenceBar = ({ value }) => {
    const val = typeof value === 'number' && Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 50;
    return (
      <div className="confidence-bar w-24">
        <div
          className={`h-full rounded-full transition-all ${
            val >= 70 ? 'bg-green-500' : val >= 40 ? 'bg-yellow-500' : 'bg-red-500'
          }`}
          style={{ width: `${val}%` }}
        />
      </div>
    );
  };

  const IndicatorRow = ({ label, value, signal, confidence }) => (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-800/50 last:border-0 text-xs">
      <span className="text-gray-300 w-32 font-medium">{label}</span>
      <span className="font-mono text-gray-100 w-20 text-right">
        {typeof value === 'number' && Number.isFinite(value)
          ? value.toFixed(2)
          : value || '—'}
      </span>
      <div className="w-16 text-center">
        {signal && <SignalBadge signal={signal} />}
      </div>
      <div className="flex items-center gap-2 w-28 justify-end">
        {confidence != null && (
          <>
            <ConfidenceBar value={confidence} />
            <span className="text-gray-400 font-mono text-[10px] w-7 text-right">
              {Math.round(confidence)}%
            </span>
          </>
        )}
      </div>
    </div>
  );

  const Section = ({ title, indicators }) => {
    if (!Array.isArray(indicators) || indicators.length === 0) return null;
    return (
      <div className="mb-4">
        <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">
          {title}
        </h4>
        {indicators.map((ind, i) => (
          <IndicatorRow key={ind.label || i} {...ind} />
        ))}
      </div>
    );
  };

  // Safe parsing across indicator categories
  const trendIndicators = [];
  const momentumIndicators = [];
  const volatilityIndicators = [];
  const volumeIndicators = [];

  const trend = data?.trend || {};
  if (trend.ema_20) trendIndicators.push({ label: 'EMA 20', ...trend.ema_20 });
  if (trend.ema_50) trendIndicators.push({ label: 'EMA 50', ...trend.ema_50 });
  if (trend.ema_200) trendIndicators.push({ label: 'EMA 200', ...trend.ema_200 });
  if (trend.adx) trendIndicators.push({ label: 'ADX', ...trend.adx });
  if (trend.supertrend) trendIndicators.push({ label: 'Supertrend', ...trend.supertrend });

  const momentum = data?.momentum || {};
  if (momentum.rsi) momentumIndicators.push({ label: 'RSI (14)', ...momentum.rsi });
  if (momentum.macd) momentumIndicators.push({ label: 'MACD', ...momentum.macd });
  if (momentum.stochastic) momentumIndicators.push({ label: 'Stochastic', ...momentum.stochastic });
  if (momentum.cci) momentumIndicators.push({ label: 'CCI', ...momentum.cci });
  if (momentum.roc) momentumIndicators.push({ label: 'ROC', ...momentum.roc });

  const volatility = data?.volatility || {};
  if (volatility.bollinger) volatilityIndicators.push({ label: 'Bollinger %B', ...volatility.bollinger });
  if (volatility.atr) volatilityIndicators.push({ label: 'ATR', ...volatility.atr });

  const volume = data?.volume || {};
  if (volume.volume_spike) volumeIndicators.push({ label: 'Volume Spike', ...volume.volume_spike });
  if (volume.obv) volumeIndicators.push({ label: 'OBV', ...volume.obv });
  if (volume.vwap) volumeIndicators.push({ label: 'VWAP', ...volume.vwap });

  const composite = data?.composite;

  return (
    <div className="card shadow-md">
      <div className="card-header">
        <span>📉</span> Technical Indicators
      </div>

      {/* Composite signal */}
      {composite && (
        <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-gray-800/50">
          <span className="text-xs text-gray-400">Composite:</span>
          <SignalBadge signal={composite.direction || 'NEUTRAL'} />
          <ConfidenceBar value={composite.confidence || 50} />
          <span className="text-xs font-semibold text-gray-200">
            {composite.confidence || 50}% confidence
          </span>
        </div>
      )}

      {trendIndicators.length > 0 && <Section title="Trend Strength" indicators={trendIndicators} />}
      {momentumIndicators.length > 0 && <Section title="Momentum Oscillators" indicators={momentumIndicators} />}
      {volatilityIndicators.length > 0 && <Section title="Volatility Bands" indicators={volatilityIndicators} />}
      {volumeIndicators.length > 0 && <Section title="Volume & Liquidity" indicators={volumeIndicators} />}

      {/* Multi-timeframe alignment */}
      {data?.multi_timeframe && typeof data.multi_timeframe === 'object' && (
        <div className="mt-3 p-3 rounded-lg bg-gray-800/50">
          <h4 className="text-[11px] uppercase tracking-wider text-gray-500 mb-2 font-semibold">
            Multi-Timeframe Trend Confluence
          </h4>
          <div className="flex gap-4">
            {Object.entries(data.multi_timeframe).map(([tf, info]) => {
              const sig = typeof info === 'string' ? info : info?.signal || 'NEUTRAL';
              return (
                <div key={tf} className="text-center">
                  <div className="text-[11px] text-gray-400 mb-1 capitalize">{tf}</div>
                  <SignalBadge signal={sig} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
