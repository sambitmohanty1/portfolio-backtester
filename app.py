import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Import custom modules
from data_engine import load_data
from portfolio_math import calculate_daily_valuation, calculate_kpis, RISK_FREE_RATE

# --- Page Configuration ---
st.set_page_config(page_title="Real-World Portfolio Backtester", layout="wide")
st.title("📈 Real-World Portfolio Backtester")
st.markdown("Track your exact share counts, buy dates, and cost basis.")

# --- 1. Portfolio Input (CSV Upload & Interactive Table) ---
st.sidebar.header("1. Define Your Holdings")
st.sidebar.write("Upload your raw broker CSV or edit the table:")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
REQUIRED_COLUMNS = ["Ticker", "Buy Date", "Shares", "Cost Price (Original Currency)"]

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        
        # Auto-Detect Broker Export Format
        if 'AsxCode' in raw_df.columns and 'Trade Date' in raw_df.columns:
            st.sidebar.success("Broker format detected & cleaned automatically!")
            
            # Clean Tickers
            def clean_ticker(ticker):
                ticker = str(ticker).strip()
                if ticker.endswith(':US'):
                    return ticker.replace(':US', '')
                else:
                    return ticker + '.AX'
            
            # Clean Prices
            def clean_price(price):
                clean_str = str(price).replace(' USD', '').replace(',', '').strip()
                return float(clean_str)

            # Map to target format
            portfolio_df = pd.DataFrame()
            portfolio_df['Ticker'] = raw_df['AsxCode'].apply(clean_ticker)
            # Handle standard DD/MM/YYYY format from broker
            portfolio_df['Buy Date'] = pd.to_datetime(raw_df['Trade Date'], format='%d/%m/%Y').dt.date
            portfolio_df['Shares'] = raw_df['Volume']
            portfolio_df['Cost Price (Original Currency)'] = raw_df['Avg Price'].apply(clean_price)
            
        else:
            # Assume it's already in the target format
            portfolio_df = raw_df
            portfolio_df['Buy Date'] = pd.to_datetime(portfolio_df['Buy Date']).dt.date
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in portfolio_df.columns]
            if missing_cols:
                st.sidebar.error(f"CSV missing columns: {', '.join(missing_cols)}")
                portfolio_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
                
    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        portfolio_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
else:
    # Default sample data
    sample_data = {
        "Ticker": ["AAPL", "AAPL", "CBA.AX", "IVV.AX", "NVDA"],
        "Buy Date": [datetime.date(2021, 1, 4), datetime.date(2023, 6, 15), datetime.date(2022, 5, 10), datetime.date(2020, 1, 2), datetime.date(2023, 1, 5)],
        "Shares": [50, 25, 100, 200, 30],
        "Cost Price (Original Currency)": [129.41, 186.01, 101.50, 42.10, 14.25]
    }
    portfolio_df = pd.DataFrame(sample_data)

edited_portfolio = st.sidebar.data_editor(
    portfolio_df, 
    num_rows="dynamic", 
    use_container_width=True,
    column_config={
        "Buy Date": st.column_config.DateColumn("Buy Date", required=True),
        "Shares": st.column_config.NumberColumn("Shares", required=True, min_value=0.01),
        "Cost Price (Original Currency)": st.column_config.NumberColumn("Cost Price", required=True, min_value=0.01)
    }
)

# --- 2. Main Execution ---
if st.sidebar.button("Run Real-World Backtest"):
    if edited_portfolio.empty or edited_portfolio['Ticker'].isna().any():
        st.error("Please add valid tickers to your portfolio.")
    else:
        with st.spinner("Downloading historical prices & valuing portfolio..."):
            
            earliest_date = pd.to_datetime(edited_portfolio['Buy Date']).min()
            end_date = datetime.date.today()
            my_tickers = edited_portfolio['Ticker'].dropna().unique().tolist()
            
            # Fetch Data via Data Engine
            prices = load_data(my_tickers, earliest_date, end_date)
            
            # Process Math via Portfolio Logic Module
            (port_cumulative, bench_cumulative, port_returns_series, bench_daily_returns, 
             total_cost_basis_aud, current_value, total_pnl, adjusted_prices) = calculate_daily_valuation(edited_portfolio, prices, earliest_date)

            # Calculate KPIs
            total_days = len(port_returns_series)
            years = total_days / 252

            port_kpis = calculate_kpis(port_returns_series, years)
            bench_kpis = calculate_kpis(bench_daily_returns, years)

            if len(port_returns_series) > 1 and len(bench_daily_returns) > 1:
                covariance = np.cov(port_returns_series, bench_daily_returns)[0][1]
                variance = np.var(bench_daily_returns)
                beta = covariance / variance if variance > 0 else 1
            else:
                beta = 1
                
            alpha = port_kpis[1] - (RISK_FREE_RATE + beta * (bench_kpis[1] - RISK_FREE_RATE))

            # --- 3. UI Rendering ---
            st.subheader("💰 Real-World Financials (AUD)")
            fin_col1, fin_col2, fin_col3 = st.columns(3)
            fin_col1.metric("Total Invested (Cost Basis)", f"${total_cost_basis_aud:,.2f}")
            fin_col2.metric("Current Portfolio Value", f"${current_value:,.2f}")
            fin_col3.metric("Total Profit / Loss", f"${total_pnl:,.2f}", f"{(total_pnl/total_cost_basis_aud)*100:.2f}%")
            
            st.divider()

            st.subheader("📦 Holdings Summary (Average Cost Basis)")
            summary_data = []
            for ticker in my_tickers:
                ticker_rows = edited_portfolio[edited_portfolio['Ticker'] == ticker]
                total_shares = ticker_rows['Shares'].sum()
                total_cost_orig = (ticker_rows['Shares'] * ticker_rows['Cost Price (Original Currency)']).sum()
                avg_cost_orig = total_cost_orig / total_shares if total_shares > 0 else 0
                
                current_price_orig = adjusted_prices[ticker].iloc[-1]
                
                summary_data.append({
                    "Ticker": ticker,
                    "Total Shares": total_shares,
                    "Avg Cost Price (Orig Currency)": f"${avg_cost_orig:.2f}",
                    "Current Price (Orig Currency)": f"${current_price_orig:.2f}"
                })
                
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("📊 Performance vs Benchmark")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time-Weighted Return", f"{port_kpis[0]*100:.2f}%", f"vs Bench: {bench_kpis[0]*100:.2f}%")
            col2.metric("CAGR", f"{port_kpis[1]*100:.2f}%", f"vs Bench: {bench_kpis[1]*100:.2f}%")
            col3.metric("Alpha (Annual)", f"{alpha*100:.2f}%")
            col4.metric("Beta vs Benchmark", f"{beta:.2f}")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Max Drawdown", f"{port_kpis[3]*100:.2f}%")
            col6.metric("Ann. Volatility", f"{port_kpis[2]*100:.2f}%")
            col7.metric("Sharpe Ratio", f"{port_kpis[4]:.2f}")
            col8.metric("Sortino Ratio", f"{port_kpis[5]:.2f}")

            st.subheader("Relative Growth ($1 Invested): Portfolio vs 50/50 Benchmark")
            chart_data = pd.DataFrame({
                'Portfolio Return Trajectory': port_cumulative,
                '50/50 Benchmark Trajectory': bench_cumulative
            })
            st.line_chart(chart_data)
