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
            fin_col2.metric("Current Portfolio
