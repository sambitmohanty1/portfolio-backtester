import pandas as pd
import numpy as np
from data_engine import BENCHMARKS, FX_TICKER

RISK_FREE_RATE = 0.04

def calculate_daily_valuation(edited_portfolio, prices, earliest_date):
    fx_rates = prices[FX_TICKER]
    my_tickers = edited_portfolio['Ticker'].dropna().unique().tolist()
    
    # Convert US stock historical prices to AUD
    us_tickers = [t for t in my_tickers if not t.endswith('.AX')]
    for ticker in us_tickers + ['^GSPC']:
        if ticker in prices.columns:
            prices[ticker] = prices[ticker] / fx_rates

    portfolio_daily_value = pd.DataFrame(index=prices.index)
    portfolio_daily_value['Total'] = 0.0
    total_cost_basis_aud = 0.0

    # Calculate daily value for each specific lot
    for index, row in edited_portfolio.iterrows():
        ticker = row['Ticker']
        buy_date = pd.to_datetime(row['Buy Date'])
        shares = row['Shares']
        cost_price = row['Cost Price (Original Currency)']
        
        if buy_date not in prices.index:
            try:
                valid_buy_date = prices.index[prices.index > buy_date][0]
            except IndexError:
                continue 
        else:
            valid_buy_date = buy_date

        if not ticker.endswith('.AX'):
            buy_day_fx = fx_rates.loc[valid_buy_date]
            cost_basis_aud = (cost_price / buy_day_fx) * shares
        else:
            cost_basis_aud = cost_price * shares
        
        total_cost_basis_aud += cost_basis_aud
        
        daily_share_value = prices[ticker].copy()
        daily_share_value.loc[:valid_buy_date - pd.Timedelta(days=1)] = 0.0 
        daily_share_value.loc[valid_buy_date:] = daily_share_value.loc[valid_buy_date:] * shares
        
        portfolio_daily_value['Total'] += daily_share_value

    portfolio_daily_value = portfolio_daily_value[portfolio_daily_value.index >= earliest_date]
    port_returns_series = portfolio_daily_value['Total'].pct_change().dropna()
    
    current_value = portfolio_daily_value['Total'].iloc[-1]
    total_pnl = current_value - total_cost_basis_aud

    # Benchmark Calculation
    bench_prices = prices[BENCHMARKS].loc[earliest_date:].copy()
    bench_returns = bench_prices.pct_change().dropna()
    bench_weights = np.array([0.5, 0.5])
    bench_daily_returns = bench_returns.dot(bench_weights)
    bench_cumulative = (1 + bench_daily_returns).cumprod()

    port_cumulative = (1 + port_returns_series).cumprod()
    common_index = port_cumulative.index.intersection(bench_cumulative.index)
    
    return (
        port_cumulative.loc[common_index], 
        bench_cumulative.loc[common_index], 
        port_returns_series.loc[common_index], 
        bench_daily_returns.loc[common_index],
        total_cost_basis_aud,
        current_value,
        total_pnl,
        prices # Return adjusted prices for UI summaries
    )

def calculate_kpis(returns_series, total_years):
    if len(returns_series) == 0:
        return 0, 0, 0, 0, 0, 0
    
    total_return = (1 + returns_series).cumprod().iloc[-1] - 1
    cagr = (1 + total_return) ** (1 / total_years) - 1 if total_years > 0 else 0
    ann_volatility = returns_series.std() * np.sqrt(252)
    
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    sharpe = (cagr - RISK_FREE_RATE) / ann_volatility if ann_volatility > 0 else 0
    downside_std = returns_series[returns_series < 0].std() * np.sqrt(252)
    sortino = (cagr - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan
    
    return total_return, cagr, ann_volatility, max_dd, sharpe, sortino
