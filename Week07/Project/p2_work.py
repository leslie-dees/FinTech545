import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
from datetime import datetime
import numpy as np

# American options
current_date = datetime(2023, 3, 3)
dividend_payment_date = datetime(2023, 3, 15)
curr_aapl_price = 165
risk_free_rate = 0.0425

options_portfolios = pd.read_csv("Week07/Project/problem2.csv")
# Portfolio Type Underlying Holding OptionType ExpirationDate Strike CurrentPrice
portfolios = options_portfolios['Portfolio'].unique()

daily_prices = pd.read_csv("Week07/Project/DailyPrices.csv")
aapl = daily_prices[['Date', 'AAPL']]

# Calculate the log returns of AAPL
log_returns = fin.return_calculate(aapl, method="LOG")

# Demean the series so there is 0 mean
aapl_mean = log_returns["AAPL"].mean()
centered_returns = log_returns["AAPL"] - aapl_mean

norm_mean, norm_std = fin.mle_normal_distribution_one_input(centered_returns)

# Simulate AAPL returns 10 days ahead
def fit_ar1(data):
    n = len(data)
    x_t = data[:-1]
    x_t1 = data[1:]
    alpha = np.cov(x_t, x_t1)[0, 1] / np.var(x_t)
    epsilon = x_t1 - alpha * x_t
    sigma = np.std(epsilon)
    return alpha, sigma

# Created new AR(1) model due to issues with prior models
alpha, sigma = fit_ar1(centered_returns)

# predictions for the next 10 returns
num_steps = 10
predictions = np.empty(num_steps)
current_data = centered_returns.values

for tt in range(num_steps):
    epsilon_t = np.random.normal(0, sigma)
    next_return = alpha * current_data[-1] + epsilon_t
    predictions[tt] = next_return
    current_data = np.append(current_data, next_return)

# Apply those returns to AAPL price
# calculate prices from returns
def calculate_prices(initial_price, returns):
    prices = [initial_price]
    for r in returns:
        price_t = prices[-1] * (1 + r)
        prices.append(price_t)
    return prices[1:]

# calculate the next 10 prices
next_10_prices = calculate_prices(curr_aapl_price, predictions)

# Create a DataFrame for the simulated prices and dates
simulated_prices_df = pd.DataFrame({
    'Date': pd.bdate_range(start=current_date, periods=11),  # Including the current date
    'AAPL': [curr_aapl_price] + next_10_prices  # Including the current price
})

# Format the 'Date' column
simulated_prices_df['Date'] = simulated_prices_df['Date'].dt.strftime('%m/%d/%Y')

# Locate the index corresponding to '03/15/2023'
index_add = simulated_prices_df[simulated_prices_df['Date'] == '03/15/2023'].index[0]

# Add 1 to the all columns after for the dividend payment
# Date AAPL
simulated_prices_df.loc[index_add:, 'AAPL'] += 1
returns_aapl = simulated_prices_df['AAPL'].pct_change().dropna()
# Concatenate the existing daily_prices DataFrame with simulated_prices_df
combined_prices_df = pd.concat([aapl, simulated_prices_df[['Date', 'AAPL']]])

combined_prices_df['Date'] = pd.to_datetime(combined_prices_df['Date'])
combined_prices_df = combined_prices_df.sort_values(by='Date').reset_index(drop=True)

DNVaR = fin.calc_var_normal(norm_mean, norm_std)*curr_aapl_price
DNES = fin.calc_expected_shortfall_normal(norm_mean, norm_std, alpha = 0.05)*curr_aapl_price
print(f"Delta Normal VaR: ${DNVaR}")
print(f"Delta Normal ES: ${DNES}")


# Simulate returns for AAPL using Monte Carlo simulations
num_simulations = 10000
simulated_returns_aapl = np.random.choice(returns_aapl, size=(num_simulations, len(returns_aapl)), replace=True)

# Rest of the code remains the same
for portfolio in portfolios:
    portfolio_data = options_portfolios[options_portfolios['Portfolio'] == portfolio]
    portfolio_value = fin.calculate_portfolio_value_american(curr_aapl_price, portfolio_data, current_date, dividend_payment_date, risk_free_rate)

    simulated_returns_aapl = np.random.choice(returns_aapl, size=(num_simulations, len(returns_aapl)), replace=True)[0, :]

    simulated_portfolio_values = portfolio_value * np.cumprod(1 + simulated_returns_aapl, axis=0)

    simulated_portfolio_returns = np.diff(simulated_portfolio_values) / (simulated_portfolio_values[:-1]+1e-8)

    # Calculate VaR and ES for the portfolio
    portfolio_var = np.percentile(simulated_portfolio_returns, 5) * portfolio_value
    portfolio_es = np.mean(simulated_portfolio_returns[simulated_portfolio_returns <= np.percentile(simulated_portfolio_returns, 5)]) * portfolio_value
    mean_portfolio_return = np.mean(simulated_portfolio_returns) * portfolio_value

    print(f"Portfolio: {portfolio}")
    print(f"Portfolio VaR: ${portfolio_var}")
    print(f"Portfolio ES: ${portfolio_es}")
    print(f"Mean Portfolio Return Value: ${mean_portfolio_return:.5f}")
    print("------------------------")