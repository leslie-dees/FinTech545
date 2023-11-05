import pandas as pd
from datetime import datetime
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from arch import arch_model

problem3 = pd.read_csv("Week06/Project/problem3.csv")
# Portfolio Type (Option/Stock) Underlying Holding (1/-1) OptionType ExpirationDate Strike CurrentPrice

portfolios = problem3['Portfolio'].unique()

aapl_price = 170.15
curr_date = datetime.strptime("10/30/2023", "%m/%d/%Y")
risk_free = 0.0525  # 5.25%
dividend_rate = 0.0057  # 0.57%

# Function to calculate the portfolio value for a given underlying value
def calculate_portfolio_value(underlying_value, portfolio):
    portfolio_value = 0.0

    for _, asset in portfolio.iterrows():
        if asset['Type'] == 'Option':
            S = underlying_value
            X = asset['Strike']
            T = (datetime.strptime(asset['ExpirationDate'], "%m/%d/%Y") - curr_date).days / 365.0
            option_type = asset['OptionType']
            market_price = asset['CurrentPrice']
            b = risk_free - dividend_rate if option_type == 'Call' else risk_free

            result = minimize_scalar(
                lambda sigma: fin.option_price_error(sigma, S, X, T, risk_free, b, option_type, market_price),
                bounds=(0.001, 5.0)  # Adjust the bounds as needed
            )
            implied_volatility = result.x

            # Calculate the option value using implied volatility
            option_value = fin.options_price(S, X, T, implied_volatility, risk_free, b, option_type)

            # Add or subtract option value to the portfolio based on Holding (1 or -1)
            portfolio_value += asset['Holding'] * option_value
        elif asset['Type'] == 'Stock':
            # If it's a stock, just add its current price to the portfolio value
            portfolio_value += asset['Holding'] * (asset['CurrentPrice'] - underlying_value)

    return portfolio_value

underlying_values = np.arange(150.0, 190.0, 1.0)

# plt.figure(figsize=(12, 8))

# for portfolio_name in portfolios:
#     portfolio_data = problem3[problem3['Portfolio'] == portfolio_name]

#     portfolio_values = [calculate_portfolio_value(underlying_value, portfolio_data) for underlying_value in underlying_values]
    
#     plt.plot(underlying_values, portfolio_values, label=portfolio_name, marker='o')

# plt.xlabel('Underlying Value (AAPL)')
# plt.ylabel('Portfolio Value')
# plt.title('Portfolio Value vs. Underlying Value')
# plt.grid(True)
# plt.legend()  # Add legend to differentiate portfolios by color
# plt.show()


daily_prices = pd.read_csv("Week06/Project/DailyPrices.csv")
aapl = daily_prices[['Date', 'AAPL']]

# Calculate the log returns of AAPL
log_returns = fin.return_calculate(aapl, method="LOG")

# Demean the series so there is 0 mean
aapl_mean = log_returns["AAPL"].mean()
centered_returns = log_returns["AAPL"] - aapl_mean

# Function to fit an AR(1) model to data
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

for t in range(num_steps):
    epsilon_t = np.random.normal(0, sigma)
    next_return = alpha * current_data[-1] + epsilon_t
    predictions[t] = next_return
    current_data = np.append(current_data, next_return)

initial_price = 170.15

# calculate prices from returns
def calculate_prices(initial_price, returns):
    prices = [initial_price]
    for r in returns:
        price_t = prices[-1] * (1 + r)
        prices.append(price_t)
    return prices[1:]

# calculate the next 10 prices
next_10_prices = calculate_prices(initial_price, predictions)

mean_price = np.mean(next_10_prices)
std_dev_price = np.std(next_10_prices)

alpha = 0.05

z_alpha = norm.ppf(alpha)
var_price = mean_price + std_dev_price * z_alpha

# Calculate ES using the manual formula
z_alpha = norm.ppf(alpha)  # Calculate the z-score for the given alpha level
es_price_manual = mean_price - (std_dev_price * (norm.pdf(z_alpha) / alpha))

# Print the results
print("Mean Price:", mean_price)
print("VaR for Normal:", var_price)
print("Expected Shortfall (ES) for Normal (Manual):", es_price_manual)