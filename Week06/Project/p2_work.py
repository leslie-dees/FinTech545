import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
import math
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin


apple_options = pd.read_csv("Week06/Project/AAPL_Options.csv")
# Stock Expiration Type Strike Last Price

# Current AAPL price
aapl_price = 170.15
curr_date = datetime(2023, 10, 30)
risk_free = 0.0525
dividend_rate = 0.0057

# Create empty lists to store implied volatilities and strike prices for calls and puts
call_strike_prices = []
call_implied_volatilities = []
put_strike_prices = []
put_implied_volatilities = []

# Iterate through each option in apple_options
for i in range(len(apple_options)):
    S = aapl_price
    X = apple_options['Strike'][i]
    T = (datetime.strptime(apple_options['Expiration'][i], "%m/%d/%Y") - curr_date).days / 365.0
    r = risk_free
    b = risk_free - dividend_rate
    option_type = apple_options['Type'][i]
    market_price = apple_options['Last Price'][i]

    # Use optimizer to find implied volatility
    result = minimize_scalar(
        lambda sigma: fin.option_price_error(sigma, S, X, T, r, b, option_type, market_price),
        bounds=(0.001, 10.0)  # Adjust the bounds as needed
    )
    implied_volatility = result.x

    # Separate implied volatilities and strike prices for calls and puts
    if option_type == 'Call':
        call_strike_prices.append(X)
        call_implied_volatilities.append(implied_volatility)
    else:
        put_strike_prices.append(X)
        put_implied_volatilities.append(implied_volatility)

# Plot implied volatility vs. strike price for calls and puts separately
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(call_strike_prices, call_implied_volatilities, c='blue', alpha=0.5, label='Call Implied Volatility')
plt.title('Call Implied Volatility vs. Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(put_strike_prices, put_implied_volatilities, c='red', alpha=0.5, label='Put Implied Volatility')
plt.title('Put Implied Volatility vs. Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.legend()

plt.tight_layout()
plt.show()