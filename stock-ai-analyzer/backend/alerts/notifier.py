"""
Alert Notifier Module for Stock AI Analyzer.

Handles multi-channel notification broadcasts via Telegram Bot API and SMTP Email,
with automatic price alert evaluation against watchlist threshold levels.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import httpx

from backend.config import settings
from backend.data.fetcher import StockDataFetcher

logger = logging.getLogger(__name__)


class AlertNotifier:
    """
    Dispatches price alerts, trade plans, and market notifications
    via Telegram and Email with graceful configuration fallbacks.
    """

    def __init__(self):
        """Initialize AlertNotifier with credentials from application settings."""
        self.telegram_bot_token = getattr(settings, "telegram_bot_token", None)
        self.telegram_chat_id = getattr(settings, "telegram_chat_id", None)
        self.smtp_host = getattr(settings, "smtp_host", "smtp.gmail.com")
        self.smtp_port = getattr(settings, "smtp_port", 587)
        self.smtp_user = getattr(settings, "smtp_user", None)
        self.smtp_password = getattr(settings, "smtp_password", None)
        self.fetcher = StockDataFetcher

    def send_telegram(self, message: str) -> bool:
        """
        Send a notification message via the Telegram Bot API.

        Args:
            message: Message text formatted in Markdown

        Returns:
            True if delivered successfully, False otherwise.
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            logger.info("Telegram notification skipped: Bot token or chat ID is not configured.")
            return False

        # Remove placeholder dummy keys
        if "your_telegram" in self.telegram_bot_token.lower():
            logger.info("Telegram notification skipped: Placeholder bot token detected.")
            return False

        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": str(self.telegram_chat_id),
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    logger.info("Telegram message delivered successfully.")
                    return True
                else:
                    logger.warning(
                        f"Telegram API responded with HTTP {response.status_code}: {response.text}"
                    )
                    return False
        except httpx.RequestError as e:
            logger.error(f"Network error while sending Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}", exc_info=True)
            return False

    def send_email(
        self,
        subject: str,
        body: str,
        recipient: Optional[str] = None,
        is_html: bool = False,
    ) -> bool:
        """
        Send an email notification via SMTP with TLS encryption.

        Args:
            subject: Email subject line
            body: Email body text (plain text or HTML)
            recipient: Target email address (default: smtp_user)
            is_html: Whether body contains HTML markup

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.smtp_user or not self.smtp_password:
            logger.info("Email notification skipped: SMTP credentials not configured.")
            return False

        if "your_email" in self.smtp_user.lower() or "your_app_password" in self.smtp_password.lower():
            logger.info("Email notification skipped: Placeholder SMTP credentials detected.")
            return False

        target_recipient = recipient or self.smtp_user

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[Stock AI Analyzer] {subject}"
            msg["From"] = f"Stock AI Analyzer <{self.smtp_user}>"
            msg["To"] = target_recipient

            # Append disclaimer to message body
            full_body = (
                f"{body}\n\n"
                f"----------------------------------------\n"
                f"{settings.disclaimer}"
            )

            mime_subtype = "html" if is_html else "plain"
            msg.attach(MIMEText(full_body, mime_subtype, "utf-8"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10.0) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.smtp_user, [target_recipient], msg.as_string())

            logger.info(f"Email successfully delivered to {target_recipient}.")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP Authentication error: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error while sending email: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email: {e}", exc_info=True)
            return False

    def check_alerts(self, watchlist: List[Any]) -> List[Dict[str, Any]]:
        """
        Scan a watchlist of stocks and trigger notifications for breached levels:
        - Buy Zone entered
        - Stop-Loss breached
        - Target price achieved

        Args:
            watchlist: List of dicts or WatchlistItem ORM models containing:
                       symbol, alert_buy_zone, alert_stop_loss, alert_target, is_active.

        Returns:
            List of triggered alert dictionaries with execution outcomes.
        """
        triggered_alerts: List[Dict[str, Any]] = []

        if not watchlist:
            return triggered_alerts

        for item in watchlist:
            try:
                # Support both dict and ORM model
                if isinstance(item, dict):
                    symbol = item.get("symbol", "").upper().strip()
                    buy_zone = item.get("alert_buy_zone")
                    stop_loss = item.get("alert_stop_loss")
                    target = item.get("alert_target")
                    is_active = item.get("is_active", True)
                else:
                    symbol = getattr(item, "symbol", "").upper().strip()
                    buy_zone = getattr(item, "alert_buy_zone", None)
                    stop_loss = getattr(item, "alert_stop_loss", None)
                    target = getattr(item, "alert_target", None)
                    is_active = getattr(item, "is_active", True)

                if not is_active or not symbol:
                    continue

                # Fetch current market price
                df = self.fetcher.get_ohlcv(symbol, period="5d", interval="1d")
                if df.empty or len(df) == 0:
                    continue

                current_price = float(df["Close"].iloc[-1])

                # Check Condition 1: Buy Zone Entered
                if buy_zone is not None and buy_zone > 0 and current_price <= buy_zone:
                    alert = self._build_alert_payload(
                        symbol=symbol,
                        alert_type="BUY_ZONE_HIT",
                        current_price=current_price,
                        trigger_level=buy_zone,
                        condition_desc=f"Price (₹{current_price:.2f}) entered Buy Zone (<= ₹{buy_zone:.2f})",
                    )
                    self._dispatch_alert(alert)
                    triggered_alerts.append(alert)

                # Check Condition 2: Stop-Loss Hit
                if stop_loss is not None and stop_loss > 0 and current_price <= stop_loss:
                    alert = self._build_alert_payload(
                        symbol=symbol,
                        alert_type="STOP_LOSS_HIT",
                        current_price=current_price,
                        trigger_level=stop_loss,
                        condition_desc=f"Price (₹{current_price:.2f}) breached Stop-Loss (<= ₹{stop_loss:.2f})",
                    )
                    self._dispatch_alert(alert)
                    triggered_alerts.append(alert)

                # Check Condition 3: Target Hit
                if target is not None and target > 0 and current_price >= target:
                    alert = self._build_alert_payload(
                        symbol=symbol,
                        alert_type="TARGET_HIT",
                        current_price=current_price,
                        trigger_level=target,
                        condition_desc=f"Price (₹{current_price:.2f}) reached Target (>= ₹{target:.2f})",
                    )
                    self._dispatch_alert(alert)
                    triggered_alerts.append(alert)

            except Exception as e:
                logger.error(f"Error evaluating alerts for item {item}: {e}", exc_info=True)

        return triggered_alerts

    def send_trade_plan_notification(self, plan: Dict[str, Any]) -> Dict[str, bool]:
        """
        Format and broadcast a newly generated trade plan across all channels.

        Args:
            plan: Trade plan dictionary generated by TradePlanGenerator

        Returns:
            Dictionary with telegram and email delivery booleans.
        """
        symbol = plan.get("symbol", "Stock")
        direction = plan.get("direction", "NO_TRADE")
        entry = plan.get("entry_price", 0.0)
        sl = plan.get("stop_loss", 0.0)
        t1 = plan.get("target_1")
        t2 = plan.get("target_2")
        rr = plan.get("risk_reward", 0.0)
        conf = plan.get("confidence", 0.0)
        trade_type = plan.get("trade_type", "SWING")
        summary = plan.get("summary", "")

        subject = f"Trade Plan: {symbol} - {direction} ({trade_type})"

        message = (
            f"⚡ *Stock AI Analyzer — Trade Plan Alert*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *Symbol:* `{symbol}` ({trade_type})\n"
            f"🧭 *Direction:* *{direction}*\n"
            f"🎯 *Entry:* ₹{entry:.2f}\n"
            f"🛑 *Stop-Loss:* ₹{sl:.2f}\n"
            f"🏁 *Target 1:* ₹{t1:.2f}\n" if t1 else ""
            f"🏁 *Target 2:* ₹{t2:.2f}\n" if t2 else ""
            f"⚖️ *Risk-Reward:* 1:{rr:.1f}\n"
            f"📊 *Confidence Score:* {conf:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 *Summary:* {summary}\n\n"
            f"⚠️ *Disclaimer:* {settings.disclaimer}"
        )

        telegram_ok = self.send_telegram(message)
        email_ok = self.send_email(subject, message)

        return {"telegram": telegram_ok, "email": email_ok}

    def _build_alert_payload(
        self,
        symbol: str,
        alert_type: str,
        current_price: float,
        trigger_level: float,
        condition_desc: str,
    ) -> Dict[str, Any]:
        """Construct structured alert dictionary with confidence score."""
        icons = {
            "BUY_ZONE_HIT": "🎯",
            "STOP_LOSS_HIT": "🛑",
            "TARGET_HIT": "🚀",
        }
        icon = icons.get(alert_type, "🔔")

        msg = (
            f"{icon} *Stock Alert: {symbol}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Condition: *{alert_type.replace('_', ' ')}*\n"
            f"Current Price: *₹{current_price:.2f}*\n"
            f"Trigger Threshold: *₹{trigger_level:.2f}*\n"
            f"Details: {condition_desc}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ {settings.disclaimer}"
        )

        return {
            "symbol": symbol,
            "alert_type": alert_type,
            "current_price": round(current_price, 2),
            "trigger_level": round(trigger_level, 2),
            "condition": condition_desc,
            "timestamp": datetime.now().isoformat(),
            "message": msg,
            "confidence": 100.0,
            "notified_telegram": False,
            "notified_email": False,
            "disclaimer": settings.disclaimer,
        }

    def _dispatch_alert(self, alert_payload: Dict[str, Any]) -> None:
        """Dispatch an individual triggered alert via Telegram and Email."""
        msg = alert_payload["message"]
        subject = f"{alert_payload['alert_type']}: {alert_payload['symbol']} at ₹{alert_payload['current_price']}"

        alert_payload["notified_telegram"] = self.send_telegram(msg)
        alert_payload["notified_email"] = self.send_email(subject, msg)
