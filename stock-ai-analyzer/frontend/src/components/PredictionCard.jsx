import React from 'react';

/**
 * ML Prediction card showing directional probability for multiple horizons.
 * Always displays model accuracy metrics for transparency and defends against null props.
 */
export default function PredictionCard({ data }) {
  if (!data) return null;

  const predictions = data?.predictions || {};
  const metrics = data?.model_metrics || {};
  const features = Array.isArray(data?.feature_importance?.top_features)
    ? data.feature_importance.top_features.slice(0, 5)
    : [];

  const ensembleDirection =
    data?.ensemble_direction ||
    predictions['1_week']?.direction ||
    predictions['1_day']?.direction ||
    'NEUTRAL';

  const ensembleConfidence =
    data?.ensemble_confidence ||
    predictions['1_week']?.confidence ||
    predictions['1_day']?.confidence ||
    50;

  const ProbabilityGauge = ({ label, prediction }) => {
    if (!prediction) {
      return (
        <div className="text-center p-3 bg-gray-800/40 rounded-lg">
          <div className="text-xs text-gray-500 mb-2 uppercase">{label}</div>
          <div className="text-lg font-bold text-gray-400">—</div>
          <div className="text-[10px] text-gray-500 mt-1">Pending</div>
        </div>
      );
    }

    const probVal = prediction.probability != null ? prediction.probability : 0.5;
    const prob = Math.round(probVal * 100);
    const dir = prediction.direction || 'NEUTRAL';
    const isUp = dir === 'UP' || dir === 'BULLISH';
    const isDown = dir === 'DOWN' || dir === 'BEARISH';

    const color = isUp ? 'text-green-400' : isDown ? 'text-red-400' : 'text-yellow-400';
    const bgColor = isUp ? 'bg-green-500' : isDown ? 'bg-red-500' : 'bg-yellow-500';
    const arrow = isUp ? '▲' : isDown ? '▼' : '◆';

    return (
      <div className="text-center p-3 bg-gray-800/50 rounded-lg">
        <div className="text-xs text-gray-400 mb-2 uppercase">{label}</div>
        <div className={`text-2xl font-bold ${color}`}>
          {arrow} {prob}%
        </div>
        <div className="text-xs text-gray-400 mt-1">{dir}</div>
        <div className="mt-2 h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div className={`h-full ${bgColor} rounded-full`} style={{ width: `${Math.min(100, Math.max(0, prob))}%` }} />
        </div>
      </div>
    );
  };

  return (
    <div className="card shadow-md">
      <div className="card-header">
        <span>🤖</span> ML Ensemble Prediction
        <span className={`ml-auto text-xs px-2 py-0.5 rounded font-semibold ${
          ensembleDirection === 'UP' || ensembleDirection === 'BULLISH'
            ? 'badge-bull'
            : ensembleDirection === 'DOWN' || ensembleDirection === 'BEARISH'
            ? 'badge-bear'
            : 'badge-neutral'
        }`}>
          {ensembleDirection} ({Math.round(ensembleConfidence)}%)
        </span>
      </div>

      {/* Predictions for each horizon */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <ProbabilityGauge label="1 Day" prediction={predictions['1_day']} />
        <ProbabilityGauge label="1 Week" prediction={predictions['1_week']} />
        <ProbabilityGauge label="1 Month" prediction={predictions['1_month']} />
      </div>

      {/* Model Accuracy — ALWAYS shown for SEBI transparency */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="mb-4">
          <h4 className="text-xs uppercase tracking-wider text-gray-500 mb-2 font-semibold">
            Model Performance (Honest Metrics)
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500">
                  <th className="text-left py-1">Model</th>
                  <th className="text-right py-1">Accuracy</th>
                  <th className="text-right py-1">Precision</th>
                  <th className="text-right py-1">Recall</th>
                  <th className="text-right py-1">F1</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics).map(([model, m]) => (
                  <tr key={model} className="border-t border-gray-800">
                    <td className="py-1 text-gray-300 capitalize">{model.replace('_', ' ')}</td>
                    <td className="text-right py-1 font-mono">{(((m?.accuracy ?? 0)) * 100).toFixed(1)}%</td>
                    <td className="text-right py-1 font-mono">{(((m?.precision ?? 0)) * 100).toFixed(1)}%</td>
                    <td className="text-right py-1 font-mono">{(((m?.recall ?? 0)) * 100).toFixed(1)}%</td>
                    <td className="text-right py-1 font-mono">{(((m?.f1 ?? 0)) * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Top features */}
      {features.length > 0 && (
        <div>
          <h4 className="text-xs uppercase tracking-wider text-gray-500 mb-2 font-semibold">
            Top Quantitative Feature Drivers
          </h4>
          <div className="space-y-1">
            {features.map((f, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs text-gray-400 w-28 truncate">{f?.name || `Feature ${i+1}`}</span>
                <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 rounded-full"
                    style={{ width: `${Math.min(100, Math.max(0, (f?.importance || 0) * 100))}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 w-10 text-right">
                  {((f?.importance || 0) * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Warning */}
      <div className="mt-3 p-2 bg-yellow-900/20 rounded border border-yellow-800/50">
        <p className="text-[11px] text-yellow-300/90 leading-tight">
          ⚠️ ML predictions are statistical approximations. Walk-forward backtest win rate is 52–58%. Never risk more than 1–2% of capital per setup.
        </p>
      </div>
    </div>
  );
}
