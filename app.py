import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Region Backtester", layout="wide")
st.title("📈 Multi-Region Portfolio Backtester")

# --- 1. Define Universe & Parameters ---
US_STOCKS = ['AAPL', 'AMZN', 'AVGO', 'GOOGL', 'MA', 'META', 'MSFT', 'MU', 'NOW', 'NVDA', 'TSLA', 'TTD', 'UNH', 'VST', 'WMT']
ASX_STOCKS = ['ANZ.AX', 'CBA.AX', 'CSL.AX', 'HUB.AX', 'MPL.AX', 'MVW.AX', 'NXT.AX', 'PDN.AX', 'PME.AX', 'PPT.AX', 'RMD.AX', 'STO.AX', 'WTC.AX', 'XRO.AX']
ETFS = ['CRYP.AX', 'GOLD.AX', 'HACK.AX', 'IVV.AX', 'VAS.AX', 'VGS.AX']

ALL_TICKERS = US_STOCKS + ASX_STOCKS + ETFS
BENCHMARKS = ['^GSPC', '^AXJO']
FX_TICKER = 'AUDUSD=X'
RISK_FREE_RATE = 0.04

# Sidebar for user inputs
st.sidebar.header("Backtest Parameters")
start_date_input = st.sidebar.date_input("Start Date", datetime.date(2020, 1, 1))
end_date_input = st.sidebar.date_input("End Date", datetime.date.today())

# --- 2. Data Engine (Cached for performance) ---
@st.cache_data
def load_data(start_date, end_date):
    tickers_to_download = ALL_TICKERS + BENCHMARKS + [FX_TICKER]
    data = yf.download(tickers_to_download, start=start_date, end=end_date)
    
    # Use 'Close' instead of 'Adj Close' due to yfinance updates
    prices = data['Close'].copy()
    fx_rates = prices[FX_TICKER]
    
    # Convert US assets and S&P 500 to AUD
    for ticker in US_STOCKS + ['^GSPC']:
        prices[ticker] = prices[ticker] / fx_rates
        
    prices.dropna(how='all', inplace=True)
    
    # Use direct ffill() method for modern Pandas compatibility
    prices.ffill(inplace=True)
    return prices

if st.sidebar.button("Run Backtest"):
    with st.spinner("Downloading historical data & calculating..."):
        prices = load_data(start_date_input, end_date_input)

        # --- 3. Execution Logic ---
        portfolio_assets = prices[ALL_TICKERS]
        daily_returns = portfolio_assets.pct_change().dropna()

        num_assets = len(ALL_TICKERS)
        target_weights = np.repeat(1 / num_assets, num_assets)
        
        # Use 'QE' (Quarter End) instead of 'Q' for modern Pandas
        rebalance_dates = daily_returns.resample('QE').last().index

        portfolio_daily_returns = []
        current_weights = target_weights.copy()

        for date, returns in daily_returns.iterrows():
            if date in rebalance_dates:
                current_weights = target_weights.copy()
            
            daily_port_ret = np.dot(current_weights, returns)
            portfolio_daily_returns.append(daily_port_ret)
            
            current_weights = current_weights * (1 + returns)
            current_weights = current_weights / current_weights.sum() 

        port_returns_series = pd.Series(portfolio_daily_returns, index=daily_returns.index)
        port_cumulative = (1 + port_returns_series).cumprod()

        # --- 4. Benchmark Calculation ---
        bench_returns = prices[BENCHMARKS].pct_change().dropna()
        bench_weights = np.array([0.5, 0.5])
        bench_daily_returns = bench_returns.dot(bench_weights)
        bench_cumulative = (1 + bench_daily_returns).cumprod()

        # --- 5. Performance Dashboard Calculations ---
        def calculate_kpis(returns_series, total_years):
            total_return = (1 + returns_series).cumprod().iloc[-1] - 1
            cagr = (1 + total_return) ** (1 / total_years) - 1
            ann_volatility = returns_series.std() * np.sqrt(252)
            
            cumulative = (1 + returns_series).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()
