"""
Indian Equity Capital Gains Tax & Statutory Charges Calculator.
Calculates STCG (20%) vs LTCG (12.5% above ₹1.25L exemption) with 4% cess,
STT, brokerage, exchange turnover fees, SEBI charges, stamp duty, and GST
as per post-Union Budget 2024 tax provisions.
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
import math

from backend.config_tax import TAX_CONFIG


class CapitalGainsTaxCalculator:
    """
    Computes accurate statutory Indian capital gains tax and transaction friction.
    Provides comparison between exiting short-term vs holding for long-term (>12 months).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or TAX_CONFIG

    def calculate(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int,
        buy_date: Any,
        sell_date: Optional[Any] = None,
        trade_type: str = "DELIVERY",
        brokerage_per_order: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate full transaction charges, capital gains tax, and net post-tax profit.

        Args:
            buy_price: Purchase price per share in INR
            sell_price: Sale price per share in INR
            quantity: Number of shares traded
            buy_date: Date of purchase (YYYY-MM-DD or date object)
            sell_date: Date of sale (YYYY-MM-DD or date object, defaults to today)
            trade_type: "DELIVERY" (CNC) or "INTRADAY" (MIS)
            brokerage_per_order: Brokerage fee per executed order in INR (default: ₹20)

        Returns:
            Structured dictionary with gross profit, itemized charges, taxable gains,
            tax breakdown, and LTCG comparison savings.
        """
        buy_price = float(buy_price)
        sell_price = float(sell_price)
        quantity = int(quantity)
        trade_type = trade_type.upper().strip()
        brokerage_fee = (
            float(brokerage_per_order)
            if brokerage_per_order is not None
            else self.cfg.get("default_flat_brokerage_per_order", 20.0)
        )

        # Parse dates
        if isinstance(buy_date, str):
            b_dt = datetime.strptime(buy_date[:10], "%Y-%m-%d").date()
        elif isinstance(buy_date, (datetime, date)):
            b_dt = buy_date if isinstance(buy_date, date) else buy_date.date()
        else:
            b_dt = date.today()

        if sell_date:
            if isinstance(sell_date, str):
                s_dt = datetime.strptime(sell_date[:10], "%Y-%m-%d").date()
            elif isinstance(sell_date, (datetime, date)):
                s_dt = sell_date if isinstance(sell_date, date) else sell_date.date()
            else:
                s_dt = date.today()
        else:
            s_dt = date.today()

        holding_days = max(0, (s_dt - b_dt).days)
        is_long_term = trade_type == "DELIVERY" and holding_days >= 365

        # 1. Turnover & Gross Profit
        buy_turnover = buy_price * quantity
        sell_turnover = sell_price * quantity
        total_turnover = buy_turnover + sell_turnover
        gross_profit = (sell_price - buy_price) * quantity

        # 2. Itemized Transaction Charges
        # Brokerage (2 orders: buy + sell)
        brokerage_total = brokerage_fee * 2.0

        # Securities Transaction Tax (STT)
        if trade_type == "INTRADAY":
            stt = sell_turnover * self.cfg.get("stt_intraday_sell_rate", 0.00025)
        else:
            stt = (buy_turnover * self.cfg.get("stt_delivery_buy_rate", 0.001)) + (
                sell_turnover * self.cfg.get("stt_delivery_sell_rate", 0.001)
            )

        # Exchange turnover charges
        exchange_charges = total_turnover * self.cfg.get("exchange_turnover_rate", 0.0000325)

        # SEBI turnover fees
        sebi_charges = total_turnover * self.cfg.get("sebi_turnover_rate", 0.000001)

        # Stamp duty (on purchase only)
        if trade_type == "INTRADAY":
            stamp_duty = buy_turnover * self.cfg.get("stamp_duty_intraday_buy", 0.00003)
        else:
            stamp_duty = buy_turnover * self.cfg.get("stamp_duty_delivery_buy", 0.00015)

        # GST: 18% on (Brokerage + Exchange turnover charges + SEBI fees)
        gst_taxable_services = brokerage_total + exchange_charges + sebi_charges
        gst = gst_taxable_services * self.cfg.get("gst_rate", 0.18)

        total_charges = brokerage_total + stt + exchange_charges + sebi_charges + stamp_duty + gst

        # 3. Net Capital Gain (before income tax)
        net_capital_gain = gross_profit - total_charges

        # 4. Actual Tax Computation
        stcg_rate = self.cfg.get("stcg_rate", 0.20)
        ltcg_rate = self.cfg.get("ltcg_rate", 0.125)
        cess_rate = self.cfg.get("cess_rate", 0.04)
        ltcg_exemption = self.cfg.get("ltcg_exemption_limit", 125000.0)

        if net_capital_gain <= 0:
            tax_type = "CAPITAL_LOSS"
            base_tax = 0.0
            cess = 0.0
            total_tax = 0.0
            taxable_gain = 0.0
        elif is_long_term:
            tax_type = "LTCG_SEC_112A"
            taxable_gain = max(0.0, net_capital_gain - ltcg_exemption)
            base_tax = taxable_gain * ltcg_rate
            cess = base_tax * cess_rate
            total_tax = base_tax + cess
        else:
            tax_type = "STCG_SEC_111A"
            taxable_gain = net_capital_gain
            base_tax = taxable_gain * stcg_rate
            cess = base_tax * cess_rate
            total_tax = base_tax + cess

        net_profit_after_tax = net_capital_gain - total_tax

        # 5. Comparative LTCG simulation: If held for >= 12 months
        hypothetical_ltcg_taxable = max(0.0, net_capital_gain - ltcg_exemption) if net_capital_gain > 0 else 0.0
        hypothetical_ltcg_base = hypothetical_ltcg_taxable * ltcg_rate
        hypothetical_ltcg_total = hypothetical_ltcg_base * (1.0 + cess_rate)
        tax_saved_by_holding = max(0.0, total_tax - hypothetical_ltcg_total) if not is_long_term else 0.0

        return {
            "summary": {
                "trade_type": trade_type,
                "holding_days": holding_days,
                "classification": "Long-Term (>=12M)" if is_long_term else "Short-Term (<12M)",
                "tax_category": tax_type,
                "gross_profit": round(gross_profit, 2),
                "total_charges": round(total_charges, 2),
                "net_gain_before_tax": round(net_capital_gain, 2),
                "total_tax_liability": round(total_tax, 2),
                "net_profit_in_hand": round(net_profit_after_tax, 2),
                "roi_pct": round((net_profit_after_tax / buy_turnover) * 100, 2) if buy_turnover > 0 else 0.0,
            },
            "charges_breakdown": {
                "brokerage": round(brokerage_total, 2),
                "stt": round(stt, 2),
                "exchange_turnover_charges": round(exchange_charges, 2),
                "sebi_turnover_charges": round(sebi_charges, 2),
                "stamp_duty": round(stamp_duty, 2),
                "gst_18_pct": round(gst, 2),
                "total": round(total_charges, 2),
            },
            "tax_breakdown": {
                "tax_section": "112A (LTCG)" if is_long_term else "111A (STCG)",
                "applicable_rate_pct": (ltcg_rate if is_long_term else stcg_rate) * 100,
                "exemption_claimed": min(net_capital_gain, ltcg_exemption) if is_long_term and net_capital_gain > 0 else 0.0,
                "taxable_gain": round(taxable_gain, 2),
                "base_tax": round(base_tax, 2),
                "cess_4_pct": round(cess, 2),
                "total_tax": round(total_tax, 2),
                "effective_tax_rate_pct": round((total_tax / net_capital_gain * 100), 2) if net_capital_gain > 0 else 0.0,
            },
            "holding_period_comparison": {
                "sold_today": {
                    "holding_days": holding_days,
                    "tax_type": tax_type,
                    "tax_amount": round(total_tax, 2),
                    "net_in_hand": round(net_profit_after_tax, 2),
                },
                "held_over_12_months": {
                    "holding_days": 365,
                    "tax_type": "LTCG_SEC_112A",
                    "exemption_limit": ltcg_exemption,
                    "tax_amount": round(hypothetical_ltcg_total, 2),
                    "net_in_hand": round(net_capital_gain - hypothetical_ltcg_total, 2),
                },
                "tax_savings_if_held": round(tax_saved_by_holding, 2),
            },
            "budget_version": self.cfg.get("effective_budget_year", "2024-2025"),
            "advisory_note": self.cfg.get("advisory_note", ""),
        }
