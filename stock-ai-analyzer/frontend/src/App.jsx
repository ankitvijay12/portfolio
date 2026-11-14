import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Disclaimer from './components/Disclaimer';
import ErrorBoundary from './components/ErrorBoundary';
import Dashboard from './pages/Dashboard';

// Lazy-load secondary pages for faster initial load
const ShortTermPage = lazy(() => import('./pages/ShortTermPage'));
const Analysis = lazy(() => import('./pages/Analysis'));
const ScreenerPage = lazy(() => import('./pages/ScreenerPage'));
const Backtest = lazy(() => import('./pages/Backtest'));

function PageLoader() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] space-y-3">
      <div className="spinner" />
      <p className="text-xs text-gray-400 font-medium">Loading page modules...</p>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary name="Application Root" title="Stock AI Analyzer Encountered An Issue">
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/short-term" element={<ShortTermPage />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/screener" element={<ScreenerPage />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
          <Disclaimer />
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
