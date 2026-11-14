"""
NSE stock symbols, sector mapping, and Nifty 50 / Nifty 500 constituents.
Used for symbol lookup, sector analysis, and screener.
"""

from typing import Dict, Any

# Top NSE stocks with sector mapping (representative subset — extend as needed)
NSE_STOCKS: Dict[str, Dict[str, str]] = {
    # Nifty 50 constituents (major ones)
    "RELIANCE": {"name": "Reliance Industries Ltd", "sector": "Energy", "industry": "Oil & Gas Refining"},
    "TCS": {"name": "Tata Consultancy Services", "sector": "Technology", "industry": "IT Services"},
    "HDFCBANK": {"name": "HDFC Bank Ltd", "sector": "Financial Services", "industry": "Banking"},
    "INFY": {"name": "Infosys Ltd", "sector": "Technology", "industry": "IT Services"},
    "ICICIBANK": {"name": "ICICI Bank Ltd", "sector": "Financial Services", "industry": "Banking"},
    "HINDUNILVR": {"name": "Hindustan Unilever Ltd", "sector": "FMCG", "industry": "Personal Products"},
    "SBIN": {"name": "State Bank of India", "sector": "Financial Services", "industry": "Banking"},
    "BHARTIARTL": {"name": "Bharti Airtel Ltd", "sector": "Telecom", "industry": "Telecom Services"},
    "ITC": {"name": "ITC Ltd", "sector": "FMCG", "industry": "Tobacco & FMCG"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank Ltd", "sector": "Financial Services", "industry": "Banking"},
    "LT": {"name": "Larsen & Toubro Ltd", "sector": "Industrials", "industry": "Construction & Engineering"},
    "AXISBANK": {"name": "Axis Bank Ltd", "sector": "Financial Services", "industry": "Banking"},
    "ASIANPAINT": {"name": "Asian Paints Ltd", "sector": "Consumer Discretionary", "industry": "Paints"},
    "MARUTI": {"name": "Maruti Suzuki India Ltd", "sector": "Automobile", "industry": "Passenger Vehicles"},
    "SUNPHARMA": {"name": "Sun Pharmaceutical Industries", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "TITAN": {"name": "Titan Company Ltd", "sector": "Consumer Discretionary", "industry": "Jewellery & Watches"},
    "BAJFINANCE": {"name": "Bajaj Finance Ltd", "sector": "Financial Services", "industry": "NBFC"},
    "BAJFINSV": {"name": "Bajaj Finserv Ltd", "sector": "Financial Services", "industry": "Financial Services"},
    "WIPRO": {"name": "Wipro Ltd", "sector": "Technology", "industry": "IT Services"},
    "HCLTECH": {"name": "HCL Technologies Ltd", "sector": "Technology", "industry": "IT Services"},
    "NESTLEIND": {"name": "Nestle India Ltd", "sector": "FMCG", "industry": "Food Products"},
    "ULTRACEMCO": {"name": "UltraTech Cement Ltd", "sector": "Materials", "industry": "Cement"},
    "TATAMOTORS": {"name": "Tata Motors Ltd", "sector": "Automobile", "industry": "Commercial Vehicles"},
    "TATASTEEL": {"name": "Tata Steel Ltd", "sector": "Materials", "industry": "Steel"},
    "POWERGRID": {"name": "Power Grid Corporation", "sector": "Utilities", "industry": "Electric Utilities"},
    "NTPC": {"name": "NTPC Ltd", "sector": "Utilities", "industry": "Power Generation"},
    "M&M": {"name": "Mahindra & Mahindra Ltd", "sector": "Automobile", "industry": "Farm & Utility Vehicles"},
    "ONGC": {"name": "Oil & Natural Gas Corp", "sector": "Energy", "industry": "Oil & Gas Exploration"},
    "JSWSTEEL": {"name": "JSW Steel Ltd", "sector": "Materials", "industry": "Steel"},
    "ADANIENT": {"name": "Adani Enterprises Ltd", "sector": "Industrials", "industry": "Conglomerate"},
    "ADANIPORTS": {"name": "Adani Ports & SEZ", "sector": "Industrials", "industry": "Port Services"},
    "TECHM": {"name": "Tech Mahindra Ltd", "sector": "Technology", "industry": "IT Services"},
    "COALINDIA": {"name": "Coal India Ltd", "sector": "Energy", "industry": "Mining"},
    "BPCL": {"name": "Bharat Petroleum Corp", "sector": "Energy", "industry": "Oil & Gas Refining"},
    "GRASIM": {"name": "Grasim Industries Ltd", "sector": "Materials", "industry": "Cement & Textiles"},
    "DIVISLAB": {"name": "Divi's Laboratories Ltd", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "DRREDDY": {"name": "Dr. Reddy's Laboratories", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "CIPLA": {"name": "Cipla Ltd", "sector": "Healthcare", "industry": "Pharmaceuticals"},
    "EICHERMOT": {"name": "Eicher Motors Ltd", "sector": "Automobile", "industry": "Two Wheelers"},
    "APOLLOHOSP": {"name": "Apollo Hospitals Enterprise", "sector": "Healthcare", "industry": "Hospitals"},
    "HEROMOTOCO": {"name": "Hero MotoCorp Ltd", "sector": "Automobile", "industry": "Two Wheelers"},
    "BRITANNIA": {"name": "Britannia Industries Ltd", "sector": "FMCG", "industry": "Food Products"},
    "INDUSINDBK": {"name": "IndusInd Bank Ltd", "sector": "Financial Services", "industry": "Banking"},
    "SBILIFE": {"name": "SBI Life Insurance", "sector": "Financial Services", "industry": "Life Insurance"},
    "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Financial Services", "industry": "Life Insurance"},
    "TATACONSUM": {"name": "Tata Consumer Products", "sector": "FMCG", "industry": "Food & Beverages"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto Ltd", "sector": "Automobile", "industry": "Two Wheelers"},
    "HINDALCO": {"name": "Hindalco Industries Ltd", "sector": "Materials", "industry": "Aluminium"},
    "UPL": {"name": "UPL Ltd", "sector": "Materials", "industry": "Agrochemicals"},
    "WIPRO": {"name": "Wipro Ltd", "sector": "Technology", "industry": "IT Services"},

    # Additional popular stocks
    "ZOMATO": {"name": "Zomato Ltd", "sector": "Consumer Discretionary", "industry": "Food Delivery"},
    "PAYTM": {"name": "One 97 Communications (Paytm)", "sector": "Technology", "industry": "Fintech"},
    "NYKAA": {"name": "FSN E-Commerce (Nykaa)", "sector": "Consumer Discretionary", "industry": "E-Commerce"},
    "IRCTC": {"name": "Indian Railway Catering & Tourism", "sector": "Industrials", "industry": "Tourism"},
    "DMART": {"name": "Avenue Supermarts (DMart)", "sector": "Consumer Discretionary", "industry": "Retail"},
    "PIDILITIND": {"name": "Pidilite Industries Ltd", "sector": "Materials", "industry": "Adhesives"},
    "HAVELLS": {"name": "Havells India Ltd", "sector": "Consumer Discretionary", "industry": "Electrical Equipment"},
    "BANKBARODA": {"name": "Bank of Baroda", "sector": "Financial Services", "industry": "Banking"},
    "PNB": {"name": "Punjab National Bank", "sector": "Financial Services", "industry": "Banking"},
    "VEDL": {"name": "Vedanta Ltd", "sector": "Materials", "industry": "Mining"},
    "TATAPOWER": {"name": "Tata Power Company", "sector": "Utilities", "industry": "Electric Utilities"},
    "ADANIGREEN": {"name": "Adani Green Energy", "sector": "Utilities", "industry": "Renewable Energy"},
    "HAL": {"name": "Hindustan Aeronautics", "sector": "Industrials", "industry": "Aerospace & Defence"},
    "BEL": {"name": "Bharat Electronics Ltd", "sector": "Industrials", "industry": "Defence Electronics"},
    "IDFCFIRSTB": {"name": "IDFC First Bank", "sector": "Financial Services", "industry": "Banking"},
}

# Sector indices for relative strength
SECTOR_INDICES = {
    "Banking": "^NSEBANK",
    "Technology": "^CNXIT",
    "Financial Services": "^CNXFIN",
    "FMCG": "^CNXFMCG",
    "Automobile": "^CNXAUTO",
    "Healthcare": "^CNXPHARMA",
    "Energy": "^CNXENERGY",
    "Materials": "^CNXMETAL",
    "Industrials": "^CNXINFRA",
    "Consumer Discretionary": "^CNXREALTY",
}

# Nifty 50 symbols for index comparison
NIFTY50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "BAJFINSV", "WIPRO", "HCLTECH",
    "NESTLEIND", "ULTRACEMCO", "TATAMOTORS", "TATASTEEL", "POWERGRID",
    "NTPC", "M&M", "ONGC", "JSWSTEEL", "ADANIENT",
    "ADANIPORTS", "TECHM", "COALINDIA", "BPCL", "GRASIM",
    "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "APOLLOHOSP",
    "HEROMOTOCO", "BRITANNIA", "INDUSINDBK", "SBILIFE", "HDFCLIFE",
    "TATACONSUM", "BAJAJ-AUTO", "HINDALCO", "UPL",
]


def get_sector(symbol: str) -> str:
    """Get the sector for a given symbol."""
    stock = NSE_STOCKS.get(symbol.upper(), {})
    return stock.get("sector", "Unknown")


def get_sector_index(sector: str) -> str:
    """Get the sector index symbol for relative strength analysis."""
    return SECTOR_INDICES.get(sector, "^NSEI")


def get_stocks_by_sector(sector: str) -> list:
    """Get all stocks in a given sector."""
    return [sym for sym, info in NSE_STOCKS.items() if info.get("sector") == sector]
