import React, { useEffect, useRef, memo } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';

/**
 * Production TradingView Lightweight Charts (v4.2.3) component.
 * Features:
 * - Creates chart ONCE and updates via series.setData() (avoids flickering and canvas leaks)
 * - Strict deduplication, date sanitization, and ascending sort
 * - Zero NaN / null / Infinity values allowed
 * - Caps candles (last 300) for maximum render performance
 * - Dynamic price lines for Entry, SL, and Target levels
 * - Safe unmount with chart.remove() and DOM container cleanup
 */
function Chart({
  data = [],
  indicators = {},
  tradePlan = null,
  height = 460,
  title = '',
  loading = false,
}) {
  const chartContainerRef = useRef(null);
  const chartInstanceRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const priceLinesRef = useRef([]);

  // 1. Initialize Chart ONCE on mount
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Clean any prior children
    chartContainerRef.current.innerHTML = '';

    const containerWidth = chartContainerRef.current.clientWidth || 600;
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth,
      height,
      layout: {
        background: { color: '#0f172a' },
        textColor: '#94a3b8',
        fontSize: 12,
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#475569', width: 1, style: 2 },
        horzLine: { color: '#475569', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: '#334155',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    // Create Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });

    // Create Volume histogram series
    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartInstanceRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    // Responsive resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth || 600,
        });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstanceRef.current) {
        try {
          chartInstanceRef.current.remove();
        } catch (e) {
          console.warn('[Chart] Error during chart removal:', e);
        }
        chartInstanceRef.current = null;
        candleSeriesRef.current = null;
        volumeSeriesRef.current = null;
        priceLinesRef.current = [];
      }
    };
  }, [height]);

  // 2. Update Series Data ONCE data or tradePlan changes
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || !Array.isArray(data) || data.length === 0) {
      return;
    }

    try {
      // Deduplicate and validate
      const seenTimes = new Set();
      const validItems = [];

      for (const item of data) {
        if (!item || !item.time) continue;
        const o = Number(item.open);
        const h = Number(item.high);
        const l = Number(item.low);
        const c = Number(item.close);
        const v = Number(item.volume) || 0;

        if (
          Number.isNaN(o) ||
          Number.isNaN(h) ||
          Number.isNaN(l) ||
          Number.isNaN(c) ||
          !Number.isFinite(o) ||
          !Number.isFinite(h) ||
          !Number.isFinite(l) ||
          !Number.isFinite(c)
        ) {
          continue;
        }

        // Standardize time string (YYYY-MM-DD)
        const timeStr = typeof item.time === 'string' ? item.time.slice(0, 10) : item.time;
        if (seenTimes.has(timeStr)) continue;
        seenTimes.add(timeStr);

        validItems.push({
          time: timeStr,
          open: o,
          high: Math.max(h, o, c),
          low: Math.min(l, o, c),
          close: c,
          volume: Math.max(0, v),
        });
      }

      // Sort strictly ascending by date
      validItems.sort((a, b) => (a.time < b.time ? -1 : a.time > b.time ? 1 : 0));

      if (validItems.length === 0) return;

      // Slice to recent 300 bars for instant rendering
      const sliced = validItems.length > 300 ? validItems.slice(-300) : validItems;

      const candles = sliced.map((item) => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));

      const volumes = sliced.map((item) => ({
        time: item.time,
        value: item.volume,
        color: item.close >= item.open ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      }));

      candleSeriesRef.current.setData(candles);
      volumeSeriesRef.current.setData(volumes);

      // 3. Clear existing price lines
      if (priceLinesRef.current.length > 0) {
        for (const line of priceLinesRef.current) {
          try {
            candleSeriesRef.current.removePriceLine(line);
          } catch (e) {
            // ignore
          }
        }
        priceLinesRef.current = [];
      }

      // 4. Draw Trade Plan Lines (Entry, SL, Targets)
      if (tradePlan && tradePlan.direction !== 'NO_TRADE') {
        const lines = [];

        if (tradePlan.entry_price && Number.isFinite(tradePlan.entry_price)) {
          lines.push(
            candleSeriesRef.current.createPriceLine({
              price: tradePlan.entry_price,
              color: '#3b82f6',
              lineWidth: 2,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `Entry ₹${tradePlan.entry_price.toFixed(2)}`,
            })
          );
        }

        if (tradePlan.stop_loss && Number.isFinite(tradePlan.stop_loss)) {
          lines.push(
            candleSeriesRef.current.createPriceLine({
              price: tradePlan.stop_loss,
              color: '#ef4444',
              lineWidth: 2,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `SL ₹${tradePlan.stop_loss.toFixed(2)}`,
            })
          );
        }

        if (tradePlan.target_1 && Number.isFinite(tradePlan.target_1)) {
          lines.push(
            candleSeriesRef.current.createPriceLine({
              price: tradePlan.target_1,
              color: '#22c55e',
              lineWidth: 2,
              lineStyle: 1,
              axisLabelVisible: true,
              title: `T1 ₹${tradePlan.target_1.toFixed(2)}`,
            })
          );
        }

        if (tradePlan.target_2 && Number.isFinite(tradePlan.target_2)) {
          lines.push(
            candleSeriesRef.current.createPriceLine({
              price: tradePlan.target_2,
              color: '#10b981',
              lineWidth: 1,
              lineStyle: 2,
              axisLabelVisible: true,
              title: `T2 ₹${tradePlan.target_2.toFixed(2)}`,
            })
          );
        }

        if (tradePlan.target_3 && Number.isFinite(tradePlan.target_3)) {
          lines.push(
            candleSeriesRef.current.createPriceLine({
              price: tradePlan.target_3,
              color: '#059669',
              lineWidth: 1,
              lineStyle: 2,
              axisLabelVisible: true,
              title: `T3 ₹${tradePlan.target_3.toFixed(2)}`,
            })
          );
        }

        priceLinesRef.current = lines;
      }

      // Auto-fit content to viewport
      if (chartInstanceRef.current) {
        chartInstanceRef.current.timeScale().fitContent();
      }
    } catch (err) {
      console.error('[Chart] Error updating series data:', err);
    }
  }, [data, tradePlan]);

  const hasData = Array.isArray(data) && data.length > 0;

  return (
    <div className="card p-0 overflow-hidden relative shadow-md">
      {title && (
        <div className="px-4 py-2 border-b border-gray-800 text-sm font-medium text-gray-300 flex items-center justify-between">
          <span>{title}</span>
          {hasData && (
            <span className="text-xs text-gray-500 font-mono">
              {data.length} Candles
            </span>
          )}
        </div>
      )}

      {loading && !hasData && (
        <div className="flex flex-col items-center justify-center h-96 bg-gray-900/60 p-4 text-center">
          <div className="spinner mb-3" />
          <span className="text-xs text-gray-400">Loading interactive TradingView candles...</span>
        </div>
      )}

      {!loading && !hasData && (
        <div className="flex flex-col items-center justify-center h-96 bg-gray-900/60 p-4 text-center">
          <span className="text-sm font-semibold text-gray-300 mb-1">No Chart Data Available</span>
          <span className="text-xs text-gray-500">
            Please enter a valid NSE ticker (e.g. RELIANCE, TCS, INFY) or click a popular stock.
          </span>
        </div>
      )}

      <div
        ref={chartContainerRef}
        style={{
          height,
          display: hasData ? 'block' : 'none',
        }}
      />
    </div>
  );
}

export default memo(Chart);
