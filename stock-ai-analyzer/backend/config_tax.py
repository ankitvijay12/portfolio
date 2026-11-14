"""
Indian Capital Gains Tax & Transaction Charges Configuration.
Configured as per Income Tax Act provisions (Post Union Budget 2024 revisions).

⚠️ STATUTORY NOTE:
Tax rates, surcharge, cess, and exemption thresholds are subject to statutory amendments
under Union Budgets. Always verify current rates with the Income Tax Department or a Chartered Accountant.
"""

from typing import Dict, Any

# Indian Equity Taxation & Statutory Levies (FY 2024-25 / AY 2025-26)
TAX_CONFIG: Dict[str, Any] = {
    "effective_budget_year": "2024-2025 (Post Union Budget July 2024)",
    
    # Capital Gains Tax Rates on Listed Equities (STT Paid)
    "stcg_rate": 0.20,              # 20.0% u/s 111A (revised from 15% to 20% w.e.f 23 July 2024)
    "ltcg_rate": 0.125,             # 12.5% u/s 112A (revised from 10% to 12.5% w.e.f 23 July 2024)
    "ltcg_exemption_limit": 125000.0, # ₹1,25,000 annual exemption limit on LTCG (raised from ₹1,00,000)
    "cess_rate": 0.04,              # 4.0% Health and Education Cess on tax amount
    
    # Statutory Holding Period Thresholds (Listed Indian Equity)
    "holding_period_months_threshold": 12, # Under 12 months = STCG; 12 months or more = LTCG
    
    # Securities Transaction Tax (STT) Rates
    "stt_delivery_buy_rate": 0.001,      # 0.1% on delivery buy turnover
    "stt_delivery_sell_rate": 0.001,     # 0.1% on delivery sell turnover
    "stt_intraday_sell_rate": 0.00025,   # 0.025% on intraday sell turnover (no STT on intraday buy)
    
    # Exchange & Regulatory Turnover Levies
    "exchange_turnover_rate": 0.0000325, # NSE turnover charge: ~0.00325%
    "sebi_turnover_rate": 0.000001,      # SEBI turnover fee: ₹10 per crore (0.0001%)
    "stamp_duty_delivery_buy": 0.00015,  # Stamp duty: 0.015% on delivery buy turnover (₹1500/cr)
    "stamp_duty_intraday_buy": 0.00003,  # Stamp duty: 0.003% on intraday buy turnover (₹300/cr)
    
    # Goods and Services Tax (GST)
    "gst_rate": 0.18,                   # 18% GST applied on (Brokerage + Exchange fees + SEBI charges)
    
    # Standard Brokerage Defaults
    "default_flat_brokerage_per_order": 20.0, # Standard discount broker fee (e.g. Zerodha/Groww ₹20/order)
    
    # Advisory note
    "advisory_note": (
        "Statutory rates reflect Indian Finance Act 2024 provisions. Surcharges (if income exceeds ₹50 Lakh) "
        "and indexation benefits (withdrawn for listed equities u/s 112A) may apply. Educational estimate only; "
        "verify with the Income Tax Department or a Chartered Accountant before filing returns."
    ),
}
