import axios from 'axios';

// Single source of truth for API base URL
const API_BASE = import.meta.env.VITE_API_URL || '/api';

console.log(`[API Client] Initialized with Base URL: ${API_BASE}`);

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000, // 30s timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling & logging
api.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.config?.url} => Status ${response.status}`);
    return response;
  },
  (error) => {
    if (axios.isCancel(error)) {
      console.log('[API Request Cancelled]', error.message);
      return Promise.reject(error);
    }
    const status = error.response?.status;
    const message = error.response?.data?.detail || error.message || 'An error occurred';
    console.warn(`[API Error] ${error.config?.url} (Status: ${status}) =>`, message);
    return Promise.reject(error);
  }
);

// ============ Health & Info API ============
export const checkHealth = (config) => api.get('/health', config);
export const getDisclaimer = (config) => api.get('/disclaimer', config);

// ============ Fast Quote API (<200ms) ============
export const getQuote = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/quote`, config);

// ============ Analysis API ============
export const getFullAnalysis = (symbol, period = '1y', config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}?period=${period}`, config);

export const getTechnicalAnalysis = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/technical`, config);

export const getFundamentalAnalysis = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/fundamental`, config);

export const getPatterns = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/patterns`, config);

export const getSupportResistance = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/support-resistance`, config);

export const getOHLCV = (symbol, period = '1y', interval = '1d', config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/ohlcv?period=${period}&interval=${interval}`, config);

export const getMarketContext = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/market-context`, config);

export const getSentiment = (symbol, config) =>
  api.get(`/analysis/${encodeURIComponent(symbol)}/sentiment`, config);

// ============ Screener API (High-speed cached) ============
export const scanStocks = (symbols, config) =>
  api.get(`/screener/scan${symbols ? `?symbols=${encodeURIComponent(symbols)}` : ''}`, config);

export const getBreakouts = (config) => api.get('/screener/breakouts', config);
export const getOversold = (config) => api.get('/screener/oversold', config);
export const getOverbought = (config) => api.get('/screener/overbought', config);
export const getVolumeSpikes = (config) => api.get('/screener/volume-spike', config);
export const getStrongRS = (config) => api.get('/screener/strong-rs', config);
export const getBullishEMA = (config) => api.get('/screener/bullish-ema', config);
export const searchStocks = (query, config) =>
  api.get(`/screener/search?q=${encodeURIComponent(query)}`, config);

// ============ Trading API ============
export const getTradePlan = (symbol, tradeType = 'SWING', capital = 1000000, riskPct = 1.0, config) =>
  api.get(
    `/trading/trade-plan/${encodeURIComponent(symbol)}?trade_type=${tradeType}&capital=${capital}&risk_pct=${riskPct}`,
    config
  );

export const getAllTradePlans = (symbol, config) =>
  api.get(`/trading/trade-plan/${encodeURIComponent(symbol)}/all`, config);

export const calculatePositionSize = (data, config) =>
  api.post('/trading/position-size', data, config);

export const getWatchlist = (config) => api.get('/trading/watchlist', config);
export const addToWatchlist = (data, config) => api.post('/trading/watchlist', data, config);
export const removeFromWatchlist = (id, config) => api.delete(`/trading/watchlist/${id}`, config);

// ============ ML API ============
export const getPrediction = (symbol, config) =>
  api.get(`/predict/${encodeURIComponent(symbol)}`, config);

export const trainModel = (symbol, horizon = 5, config) =>
  api.get(`/predict/${encodeURIComponent(symbol)}/train?horizon=${horizon}`, config);

export const getFeatures = (symbol, config) =>
  api.get(`/predict/${encodeURIComponent(symbol)}/features`, config);

// ============ Backtest API ============
export const runBacktest = (data, config) => api.post('/backtest/run', data, config);
export const getStrategies = (config) => api.get('/backtest/strategies', config);
export const runAllBacktests = (symbol, config) =>
  api.get(`/backtest/run-all/${encodeURIComponent(symbol)}`, config);

// ============ Short-Term Gain API ============
export const getShortTermScan = (params = {}, config) => {
  const query = new URLSearchParams(params).toString();
  return api.get(`/short-term/scan${query ? `?${query}` : ''}`, config);
};

export const getShortTermPlan = (symbol, mode = 'SWING', capital = 1000000, riskPct = 1.0, config) =>
  api.get(
    `/short-term/plan/${encodeURIComponent(symbol)}?mode=${encodeURIComponent(mode)}&capital=${capital}&risk_pct=${riskPct}`,
    config
  );

export const getShortTermBacktestStats = (config) => api.get('/short-term/backtest-stats', config);
export const getTaxConfig = (config) => api.get('/short-term/tax-config', config);
export const calculateTax = (data, config) => api.post('/short-term/tax-calculate', data, config);
export const getJournal = (config) => api.get('/short-term/journal', config);
export const addToJournal = (data, config) => api.post('/short-term/journal', data, config);
export const updateJournal = (id, data, config) => api.patch(`/short-term/journal/${id}`, data, config);
export const deleteFromJournal = (id, config) => api.delete(`/short-term/journal/${id}`, config);
export const getPastSignals = (config) => api.get('/short-term/past-signals', config);

export default api;

