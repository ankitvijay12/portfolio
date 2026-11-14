"""
Alerts Module for Stock AI Analyzer.

Provides multi-channel notification dispatching (Telegram, Email/SMTP)
and automated watchlist price trigger monitoring.
"""

from .notifier import AlertNotifier

__all__ = ["AlertNotifier"]
