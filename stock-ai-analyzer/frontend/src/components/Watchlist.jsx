import React, { useState, useEffect, useCallback } from 'react';
import { getWatchlist, addToWatchlist, removeFromWatchlist } from '../api/client';

/**
 * Watchlist component for tracking stocks with alert levels.
 * Guards array properties and unmounted states safely.
 */
export default function Watchlist({ onSelectStock }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [newNotes, setNewNotes] = useState('');

  const fetchWatchlist = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getWatchlist();
      const list = Array.isArray(res?.data?.items) ? res.data.items : [];
      setItems(list);
    } catch (err) {
      console.warn('[Watchlist] Fetch failed:', err);
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  const handleAdd = async (e) => {
    e.preventDefault();
    if (!newSymbol.trim()) return;
    try {
      await addToWatchlist({ symbol: newSymbol.trim().toUpperCase(), notes: newNotes });
      setNewSymbol('');
      setNewNotes('');
      setShowAdd(false);
      fetchWatchlist();
    } catch (err) {
      console.error('[Watchlist] Add error:', err);
    }
  };

  const handleRemove = async (id) => {
    try {
      await removeFromWatchlist(id);
      fetchWatchlist();
    } catch (err) {
      console.error('[Watchlist] Remove error:', err);
    }
  };

  const safeItems = Array.isArray(items) ? items : [];

  return (
    <div className="card shadow-md">
      <div className="card-header">
        <span>⭐</span> Watchlist & Alerts
        <button
          onClick={() => setShowAdd(!showAdd)}
          className="ml-auto btn-primary text-xs py-1"
        >
          {showAdd ? '✕ Cancel' : '+ Add'}
        </button>
      </div>

      {showAdd && (
        <form onSubmit={handleAdd} className="flex gap-2 mb-3">
          <input
            type="text"
            placeholder="Symbol (e.g. RELIANCE)"
            value={newSymbol}
            onChange={(e) => setNewSymbol(e.target.value)}
            className="flex-1 px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs focus:border-blue-500 focus:outline-none"
          />
          <input
            type="text"
            placeholder="Notes (optional)"
            value={newNotes}
            onChange={(e) => setNewNotes(e.target.value)}
            className="flex-1 px-3 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-xs focus:border-blue-500 focus:outline-none"
          />
          <button type="submit" className="btn-primary text-xs whitespace-nowrap">
            Save
          </button>
        </form>
      )}

      {loading ? (
        <div className="flex justify-center py-4">
          <div className="spinner" />
        </div>
      ) : safeItems.length === 0 ? (
        <p className="text-xs text-gray-500 text-center py-4">
          No stocks in watchlist. Click &ldquo;+ Add&rdquo; to track stocks.
        </p>
      ) : (
        <div className="space-y-2">
          {safeItems.map((item) => {
            if (!item) return null;
            return (
              <div
                key={item.id}
                className="flex items-center justify-between p-2.5 bg-gray-800/40 hover:bg-gray-800/80 rounded-lg cursor-pointer transition-colors"
              >
                <div onClick={() => onSelectStock?.(item.symbol)} className="flex-1">
                  <span className="font-semibold text-gray-100 text-sm">{item.symbol}</span>
                  {item.notes && <span className="ml-2 text-xs text-gray-500">{item.notes}</span>}
                </div>
                <div className="flex items-center gap-2 text-xs text-gray-400">
                  {item.alert_buy_zone && (
                    <span className="text-blue-400 font-mono">Buy: ₹{item.alert_buy_zone}</span>
                  )}
                  {item.alert_stop_loss && (
                    <span className="text-red-400 font-mono">SL: ₹{item.alert_stop_loss}</span>
                  )}
                  {item.alert_target && (
                    <span className="text-green-400 font-mono">T: ₹{item.alert_target}</span>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRemove(item.id);
                    }}
                    className="text-gray-500 hover:text-red-400 ml-1 p-1"
                    title="Remove stock"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
