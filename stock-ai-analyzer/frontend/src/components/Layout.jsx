import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: '📊' },
  { path: '/short-term', label: 'Short-Term', icon: '⚡' },
  { path: '/analysis', label: 'Analysis', icon: '🔬' },
  { path: '/screener', label: 'Screener', icon: '🔍' },
  { path: '/backtest', label: 'Backtest', icon: '📈' },
];

export default function Layout({ children }) {
  const location = useLocation();
  const [searchValue, setSearchValue] = useState('');

  return (
    <div className="min-h-screen pb-12">
      {/* Top navigation */}
      <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
              <span className="text-xl">📊</span>
              <span className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Stock AI Analyzer
              </span>
              <span className="text-xs text-gray-500 ml-1">NSE/BSE</span>
            </Link>

            {/* Navigation links */}
            <div className="flex items-center gap-1">
              {NAV_ITEMS.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    location.pathname === item.path
                      ? 'bg-blue-600/20 text-blue-400'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                  }`}
                >
                  <span className="mr-1">{item.icon}</span>
                  {item.label}
                </Link>
              ))}
            </div>

            {/* Health indicator */}
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              Live
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {children}
      </main>
    </div>
  );
}
