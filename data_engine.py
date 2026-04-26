import yfinance as yf
import streamlit as st

BENCHMARKS = ['^GSPC', '^AXJO']
FX_TICKER = 'AUDUSD=X'

@st.cache_data
def load_data(tickers, start_date, end_date):
    # Ensure benchmarks and FX are included
    all_tickers_to_download = list(set(tickers + BENCHMARKS + [FX_TICKER]))
    
    # Let yfinance handle the connection natively
    data = yf.download(all_tickers_to_download, start=start_date, end=end_date)
    prices = data['Close'].copy()
    
    # Forward fill missing data for modern Pandas compatibility
    prices.ffill(inplace=True)
    return prices
