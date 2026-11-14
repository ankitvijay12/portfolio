import React from 'react';
import { useNavigate } from 'react-router-dom';
import Screener from '../components/Screener';
import ErrorBoundary from '../components/ErrorBoundary';

export default function ScreenerPage() {
  const navigate = useNavigate();

  const handleSelectStock = (symbol) => {
    navigate(`/?symbol=${symbol}`);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <span>🔍</span> NSE Stock Screener & Quantitative Scanner
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Real-time scans across Nifty 50 & 500 for breakouts, oversold bounces, volume spikes, and relative strength leaders.
          </p>
        </div>
      </div>

      <ErrorBoundary name="Dedicated Screener Page">
        <Screener onSelectStock={handleSelectStock} />
      </ErrorBoundary>
    </div>
  );
}
