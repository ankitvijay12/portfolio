import React from 'react';

/**
 * SEBI Disclaimer banner — always visible at the bottom of the page.
 */
export default function Disclaimer() {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-yellow-900/90 border-t border-yellow-700 px-4 py-2 z-50">
      <p className="text-xs text-yellow-200 text-center max-w-6xl mx-auto">
        ⚠️ <strong>DISCLAIMER:</strong> This is an educational tool for stock market analysis.
        It is NOT SEBI-registered investment advice. Past performance does not guarantee future results.
        No prediction model is 100% accurate. Always do your own research and consult a SEBI-registered
        financial advisor before making investment decisions. Trade at your own risk.
      </p>
    </div>
  );
}
