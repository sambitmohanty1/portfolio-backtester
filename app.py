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
    
    prices = data['Adj Close'].copy()
    fx_rates = prices[FX_TICKER]
    
    # Convert US assets and S&P 500 to AUD
    for ticker in US_STOCKS + ['^GSPC']:
        prices[ticker] = prices[ticker] / fx_rates
        
    prices.dropna(how='all', inplace=True)
    prices.fillna(method='ffill', inplace=True)
    return prices

if st.sidebar.button("Run Backtest"):
    with st.spinner("Downloading historical data & calculating..."):
        prices = load_data(start_date_input, end_date_input)

        # --- 3. Execution Logic ---
        portfolio_assets = prices[ALL_TICKERS]
        daily_returns = portfolio_assets.pct_change().dropna()

        num_assets = len(ALL_TICKERS)
        target_weights = np.repeat(1 / num_assets, num_assets)
        rebalance_dates = daily_returns.resample('Q').last().index

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
            
            sharpe = (cagr - RISK_FREE_RATE) / ann_volatility
            downside_std = returns_series[returns_series < 0].std() * np.sqrt(252)
            sortino = (cagr - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan
            
            return total_return, cagr, ann_volatility, max_dd, sharpe, sortino

        total_days = len(port_returns_series)
        years = total_days / 252

        port_kpis = calculate_kpis(port_returns_series, years)
        bench_kpis = calculate_kpis(bench_daily_returns, years)

        covariance = np.cov(port_returns_series, bench_daily_returns)[0][1]
        variance = np.var(bench_daily_returns)
        beta = covariance / variance
        alpha = port_kpis[1] - (RISK_FREE_RATE + beta * (bench_kpis[1] - RISK_FREE_RATE))

        # --- 6. Streamlit UI Rendering ---
        st.subheader("Performance Dashboard (AUD)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{port_kpis[0]*100:.2f}%", f"vs Bench: {bench_kpis[0]*100:.2f}%")
        col2.metric("CAGR", f"{port_kpis[1]*100:.2f}%", f"vs Bench: {bench_kpis[1]*100:.2f}%")
        col3.metric("Alpha (Annual)", f"{alpha*100:.2f}%")
        col4.metric("Beta vs Benchmark", f"{beta:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Max Drawdown", f"{port_kpis[3]*100:.2f}%")
        col6.metric("Ann. Volatility", f"{port_kpis[2]*100:.2f}%")
        col7.metric("Sharpe Ratio", f"{port_kpis[4]:.2f}")
        col8.metric("Sortino Ratio", f"{port_kpis[5]:.2f}")

        st.subheader("Cumulative Growth: Portfolio vs 50/50 Benchmark")
        chart_data = pd.DataFrame({
            'Portfolio': port_cumulative,
            '50/50 Benchmark': bench_cumulative
        })
        st.line_chart(chart_data)
else:
    st.info("Adjust your parameters in the sidebar and click 'Run Backtest' to start.")
