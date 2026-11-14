import React from 'react';

/**
 * Robust React Error Boundary component.
 * Catches JavaScript errors in child component trees, logs them,
 * and renders a polished fallback card with a visible "Retry" button
 * instead of letting an unhandled error turn the entire screen blank/black.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error(
      `[ErrorBoundary] Caught error in ${this.props.name || 'Component'}:`,
      error,
      errorInfo
    );
    this.setState({ errorInfo });
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return typeof this.props.fallback === 'function'
          ? this.props.fallback({ error: this.state.error, retry: this.handleRetry })
          : this.props.fallback;
      }

      return (
        <div className="card border border-red-800/60 bg-red-950/20 p-5 rounded-xl shadow-lg my-3">
          <div className="flex items-start gap-3">
            <span className="text-xl">⚠️</span>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-red-300">
                {this.props.title || `Something went wrong in ${this.props.name || 'this panel'}`}
              </h3>
              <p className="text-xs text-red-200/80 mt-1">
                {this.props.fallbackMessage ||
                  (this.state.error?.message ? `${this.state.error.message}` : 'An unexpected rendering error occurred.')}
              </p>

              {process.env.NODE_ENV !== 'production' && this.state.error?.stack && (
                <details className="mt-2 text-[10px] font-mono text-gray-400 bg-black/40 p-2 rounded max-h-32 overflow-auto">
                  <summary className="cursor-pointer text-gray-300 hover:text-white">Stack trace details</summary>
                  <pre className="mt-1 whitespace-pre-wrap">{this.state.error.stack}</pre>
                </details>
              )}

              <div className="mt-3 flex items-center gap-3">
                <button
                  type="button"
                  onClick={this.handleRetry}
                  className="px-3 py-1.5 bg-red-700 hover:bg-red-600 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 shadow"
                >
                  <span>🔄</span> Retry
                </button>
                <span className="text-[11px] text-gray-400">
                  Other panels remain fully active.
                </span>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
