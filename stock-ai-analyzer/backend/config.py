"""
Application configuration using Pydantic Settings.
Loads from .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./stock_analyzer.db"

    # Data APIs
    news_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None

    # Alerts
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None

    # Scheduler
    data_refresh_hour: int = 18
    data_refresh_minute: int = 0

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    debug: bool = True

    # Trading defaults
    default_capital: float = 1000000.0  # 10 Lakh INR
    default_risk_pct: float = 1.0  # 1% risk per trade
    brokerage_per_trade: float = 20.0  # ₹20 flat
    stt_rate: float = 0.001  # 0.1% STT on sell
    slippage_pct: float = 0.05  # 0.05% slippage

    # SEBI Disclaimer
    disclaimer: str = (
        "⚠️ DISCLAIMER: This is an educational tool for stock market analysis. "
        "It is NOT SEBI-registered investment advice. Past performance does not "
        "guarantee future results. No prediction model is 100% accurate. "
        "Always do your own research and consult a SEBI-registered financial "
        "advisor before making investment decisions. Trade at your own risk."
    )

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
