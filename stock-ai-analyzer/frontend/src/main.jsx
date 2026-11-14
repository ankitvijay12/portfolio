import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/global.css';

// Global error notification handler to prevent silent white/black screen crashes
function showCrashToast(message) {
  try {
    let toastContainer = document.getElementById('global-crash-toast-container');
    if (!toastContainer) {
      toastContainer = document.createElement('div');
      toastContainer.id = 'global-crash-toast-container';
      toastContainer.style.position = 'fixed';
      toastContainer.style.bottom = '16px';
      toastContainer.style.right = '16px';
      toastContainer.style.zIndex = '99999';
      toastContainer.style.display = 'flex';
      toastContainer.style.flexDirection = 'column';
      toastContainer.style.gap = '8px';
      toastContainer.style.maxWidth = '420px';
      document.body.appendChild(toastContainer);
    }

    const toast = document.createElement('div');
    toast.style.background = '#1e1b4b';
    toast.style.border = '1px solid #4338ca';
    toast.style.color = '#e0e7ff';
    toast.style.padding = '10px 14px';
    toast.style.borderRadius = '8px';
    toast.style.fontSize = '12px';
    toast.style.boxShadow = '0 10px 15px -3px rgba(0, 0, 0, 0.5)';
    toast.style.display = 'flex';
    toast.style.alignItems = 'center';
    toast.style.justifyContent = 'space-between';
    toast.style.gap = '12px';
    toast.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:16px;">ℹ️</span>
        <span>${message || 'Network or background task warning.'}</span>
      </div>
      <button style="background:none;border:none;color:#a5b4fc;cursor:pointer;font-weight:bold;font-size:14px;" onclick="this.parentElement.remove()">✕</button>
    `;

    toastContainer.appendChild(toast);
    setTimeout(() => {
      if (toast.parentElement) toast.remove();
    }, 6000);
  } catch (err) {
    console.error('[Global Error Reporter]', err);
  }
}

window.onerror = function (message, source, lineno, colno, error) {
  console.warn('[Global Uncaught Error]', { message, source, lineno, colno, error });
  // Ignore harmless resize observer and cancelled network request errors
  if (
    typeof message === 'string' &&
    (message.includes('ResizeObserver') || message.includes('canceled') || message.includes('aborted'))
  ) {
    return true;
  }
  showCrashToast(`Non-fatal warning: ${message}`);
  return true; // Prevents default browser crash
};

window.addEventListener('unhandledrejection', function (event) {
  console.warn('[Global Unhandled Promise Rejection]', event.reason);
  const reasonMsg = event.reason?.message || String(event.reason || '');
  if (
    reasonMsg.includes('canceled') ||
    reasonMsg.includes('aborted') ||
    reasonMsg.includes('ResizeObserver')
  ) {
    event.preventDefault();
    return;
  }
  showCrashToast(`Request notice: ${reasonMsg || 'Background operation timed out.'}`);
  event.preventDefault(); // Prevents unhandled promise crash
});

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
