import plotly.express as px
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

# --- 1. Portfolio Input (Dynamic CSV Upload & Interactive Table) ---
st.sidebar.header("1. Define Your Holdings")
st.sidebar.write("Upload your raw broker CSV or edit the table:")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Initialize an empty target dataframe
portfolio_df = pd.DataFrame(columns=["Ticker", "Buy Date", "Shares", "Cost Price (Original Currency)"])

if uploaded_file is not None:
    try:
        # utf-8-sig removes hidden BOM characters that break column matching
        raw_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        
        # FUZZY COLUMN MATCHER: Scan headers for keywords instead of exact string matches
        lower_cols = raw_df.columns.str.lower().str.strip()
        
        ticker_col = next((c for c, l in zip(raw_df.columns, lower_cols) if any(x in l for x in ['ticker', 'code', 'symbol', 'asx'])), None)
        date_col = next((c for c, l in zip(raw_df.columns, lower_cols) if 'date' in l), None)
        price_col = next((c for c, l in zip(raw_df.columns, lower_cols) if any(x in l for x in ['price', 'cost', 'avg'])), None)
        shares_col = next((c for c, l in zip(raw_df.columns, lower_cols) if any(x in l for x in ['share', 'qty', 'vol', 'unit'])), None)

        if not all([ticker_col, date_col, price_col]):
            st.sidebar.error(f"Could not auto-map columns. Found: Ticker({ticker_col}), Date({date_col}), Price({price_col})")
        else:
            # Clean Tickers
            def clean_ticker(t):
                t = str(t).strip().upper()
                if t.endswith(':US'): return t.replace(':US', '')
                if '.' not in t: return t + '.AX'
                return t
                
            # Clean Prices (Strip USD, AUD, $, commas)
            def clean_price(p):
                clean_str = str(p).upper().replace('USD', '').replace('AUD', '').replace('$', '').replace(',', '').strip()
                try:
                    return float(clean_str)
                except ValueError:
                    return 0.0

            # Map the dynamically found columns to the standard format the execution logic needs
            portfolio_df['Ticker'] = raw_df[ticker_col].apply(clean_ticker)
            
            # format='mixed' handles standard DD/MM/YYYY, YYYY-MM-DD, etc. seamlessly
            portfolio_df['Buy Date'] = pd.to_datetime(raw_df[date_col], format='mixed', dayfirst=True, errors='coerce').dt.date
            
            # Safely handle shares/volume
            if shares_col:
                portfolio_df['Shares'] = pd.to_numeric(raw_df[shares_col], errors='coerce').fillna(1.0)
            else:
                st.sidebar.warning("No Volume/Qty column found. Defaulting to 1 share per trade.")
                portfolio_df['Shares'] = 1.0
                
            portfolio_df['Cost Price (Original Currency)'] = raw_df[price_col].apply(clean_price)
            st.sidebar.success("CSV dynamically mapped and loaded!")
            
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {e}")
else:
    # Default sample data if no file is uploaded
    sample_data = {
        "Ticker": ["AAPL", "AAPL", "CBA.AX", "IVV.AX", "NVDA"],
        "Buy Date": [datetime.date(2021, 1, 4), datetime.date(2023, 6, 15), datetime.date(2022, 5, 10), datetime.date(2020, 1, 2), datetime.date(2023, 1, 5)],
        "Shares": [50, 25, 100, 200, 30],
        "Cost Price (Original Currency)": [129.41, 186.01, 101.50, 42.10, 14.25]
    }
    portfolio_df = pd.DataFrame(sample_data)

# Render the interactive UI table
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

            st.subheader("📦 Holdings Summary & Asset Breakdown")
            summary_data = []
            for ticker in my_tickers:
                ticker_rows = edited_portfolio[edited_portfolio['Ticker'] == ticker]
                total_shares = ticker_rows['Shares'].sum()
                total_cost_orig = (ticker_rows['Shares'] * ticker_rows['Cost Price (Original Currency)']).sum()
                avg_cost_orig = total_cost_orig / total_shares if total_shares > 0 else 0
                
                current_price_orig = adjusted_prices[ticker].iloc[-1]
                
                # Calculate individual asset return percentage
                asset_return = ((current_price_orig - avg_cost_orig) / avg_cost_orig) * 100 if avg_cost_orig > 0 else 0
                
                summary_data.append({
                    "Ticker": ticker,
                    "Total Shares": total_shares,
                    "Avg Cost Price (Orig Currency)": f"${avg_cost_orig:.2f}",
                    "Current Price (Orig Currency)": f"${current_price_orig:.2f}",
                    "Total Return (%)": asset_return
                })
                
            summary_df = pd.DataFrame(summary_data)
            
            # Display the table, formatting the return column cleanly
            st.dataframe(summary_df.style.format({"Total Return (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)

            # --- DYNAMIC PLOTLY CHART (Horizontal with Red/Green Gradient) ---
            summary_df = summary_df.sort_values('Total Return (%)', ascending=True)
            
            fig = px.bar(
                summary_df,
                x='Total Return (%)',
                y='Ticker',
                orientation='h',
                color='Total Return (%)',
                color_continuous_scale=px.colors.diverging.RdYlGn,
                color_continuous_midpoint=0,
                title="Individual Asset Performance (Drags vs Drivers)"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- PERFORMANCE VS BENCHMARK ---
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

            st.divider()

            # --- EXPORT TO CONSULTANT GEM ---
            st.subheader("🤖 Export to Consultant Gem")
            st.write("Generate a raw quant report to paste directly into your Consultant Gem for AI optimization.")
            
            if st.button("Crunch Data & Generate Report"):
                with st.spinner("Calculating Correlations, Betas, and Drawdowns..."):
                    # 1. Drawdown Date Analysis
                    peak = port_cumulative.cummax()
                    drawdown = (port_cumulative - peak) / peak
                    max_dd_date = drawdown.idxmin()

                    # 2. Individual Asset Betas
                    returns = prices[my_tickers].pct_change().dropna()
                    asset_betas = {}
                    bench_var = np.var(bench_daily_returns)
                    for t in my_tickers:
                        cov = np.cov(returns[t], bench_daily_returns)[0][1]
                        asset_betas[t] = cov / bench_var if bench_var > 0 else 1

                    # 3. Correlation Matrix (Top Pairs)
                    corr_matrix = returns.corr()
                    corr_pairs = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                    corr_pairs = corr_pairs[corr_pairs < 0.999].head(5)

                    # 4. Generate the Markdown Report
                    report_text = f"""### 📊 QUANT PORTFOLIO DATA EXPORT
*Date Generated: {datetime.date.today()}*

**1. TOP-LEVEL METRICS**
- **Time-Weighted CAGR:** {port_kpis[1]*100:.2f}%
- **Portfolio Beta:** {beta:.2f}
- **Maximum Drawdown:** {port_kpis[3]*100:.2f}% *(Hit exact bottom on: {max_dd_date.strftime('%Y-%m-%d')})*
- **Annual Alpha:** {alpha*100:.2f}%

**2. INDIVIDUAL ASSET RISK CONTRIBUTIONS (BETAS)**
*(Consultant: Identify which assets are causing the portfolio's volatility spikes)*
"""
                    for t, b in asset_betas.items():
                        report_text += f"- **{t}**: {b:.2f}\n"

                    report_text += "\n**3. HIGH CORRELATION WARNINGS**\n*(Consultant: Suggest replacement assets if portfolio lacks true diversification)*\n"
                    for pair, val in corr_pairs.items():
                        report_text += f"- **{pair[0]} & {pair[1]}**: {val:.2f} correlation\n"

                    report_text += "\n**4. ASSET RETURN DRIVERS**\n"
                    for index, row in summary_df.iterrows():
                        report_text += f"- **{row['Ticker']}**: {row['Total Return (%)']:.2f}%\n"

                    st.code(report_text, language='markdown')
                    st.success("👆 Click the 'Copy' icon in the top right of the box above, and paste it into your Consultant Gem!")
