from datetime import datetime
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
import math
from scipy.stats import norm


# Implement closed form greeks for GBSM
## In fin_package
        
# Implement finite difference derivative calculation
## In fin_package

# Compare the values between the two methods for call and put
curr_stock_price = 151.03
strike_price = 165
current_date = datetime(2022, 3, 13)
options_expiration_date = datetime(2022, 4, 15)
risk_free_rate = 0.0425
continuously_compounding_coupon = 0.0053


option_type = "call"
implied_volatility = fin.calculate_implied_volatility(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type)
function_greeks = fin.greeks(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type)
df_greeks = fin.greeks_df(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type, epsilon = 0.01)

print("Call Greeks ------------------------------------------------")
print(f"Function Delta: {function_greeks[0]}")
print(f"Discrete Delta: {df_greeks[0]}")

print(f"Function Gamma: {function_greeks[1]}")
print(f"Discrete Gamma: {df_greeks[1]}")

print(f"Function Vega: {function_greeks[2]}")
print(f"Discrete Vega: {df_greeks[2]}")

print(f"Function Theta: {function_greeks[3]}")
print(f"Discrete Theta: {df_greeks[3]}")

print(f"Function Rho: {function_greeks[4]}")
print(f"Discrete Rho: {df_greeks[4]}")

print(f"Function Carry Rho: {function_greeks[5]}")
print(f"Discrete Carry Rho: {df_greeks[5]}")

option_type = "put"
implied_volatility = fin.calculate_implied_volatility(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, 'call')
function_greeks = fin.greeks(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type)
df_greeks = fin.greeks_df(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type, epsilon = 0.01)

print("Put Greeks ------------------------------------------------")
print(f"Function Delta: {function_greeks[0]}")
print(f"Discrete Delta: {df_greeks[0]}")

print(f"Function Gamma: {function_greeks[1]}")
print(f"Discrete Gamma: {df_greeks[1]}")

print(f"Function Vega: {function_greeks[2]}")
print(f"Discrete Vega: {df_greeks[2]}")

print(f"Function Theta: {function_greeks[3]}")
print(f"Discrete Theta: {df_greeks[3]}")

print(f"Function Rho: {function_greeks[4]}")
print(f"Discrete Rho: {df_greeks[4]}")

print(f"Function Carry Rho: {function_greeks[5]}")
print(f"Discrete Carry Rho: {df_greeks[5]}")

# Implement binomial tree valuation for American options with and without discrete dividends
# Assume pays dividend on 4/11/2022 of $0.88

# separate implied volatility function to help puts converge
def calculate_implied_volatility_newton(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type, tol=1e-5, max_iter=500):
    S = curr_stock_price
    X = strike_price
    T = (options_expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuously_compounding_coupon
    b = r - q

    def calc_option_price(sigma):
        option_price = fin.options_price(S, X, T, sigma, r, b, option_type)
        return option_price

    def calc_vega(sigma):
        d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp((b - r) * T) * norm.pdf(d1) * math.sqrt(T)
        return vega

    iteration = 0
    volatility = 0.2  # Initial guess

    while iteration <= max_iter:
        option_price = calc_option_price(volatility)
        vega = calc_vega(volatility)

        if abs(option_price) < tol:
            return volatility

        volatility = volatility - option_price / vega

        iteration += 1

    raise ValueError("Implied volatility calculation did not converge")


curr_stock_price = 151.03
strike_price = 165
current_date = datetime(2022, 3, 13)
options_expiration_date = datetime(2022, 4, 15)
ttm = (options_expiration_date-current_date).days / 365
risk_free_rate = 0.0425
continuously_compounding_coupon = 0.0053
dividend_payout_date = datetime(2022, 4, 11)
num_days_dividend = (dividend_payout_date - current_date).days
num_steps = num_days_dividend+1
div_times = np.array([num_days_dividend])
div_amounts = np.array([0.88])

# Calculate value of call and put for each
option_type = 'call'
implied_volatility = calculate_implied_volatility_newton(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type)

print("Call Binomial Trees ------------------------------------")
no_div_bt = fin.binomial_tree_option_pricing_american_complete(curr_stock_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type)
print(f"No Dividend Binomial Tree: {no_div_bt}")
div_bt = fin.binomial_tree_option_pricing_american_complete(curr_stock_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type, div_amounts, div_times)
print(f"Dividend Binomial Tree: {div_bt}")

option_type = 'put'
implied_volatility = calculate_implied_volatility_newton(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type, max_iter = 20000)

print("Put Binomial Trees ------------------------------------")
no_div_bt = fin.binomial_tree_option_pricing_american_complete(curr_stock_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type)
print(f"No Dividend Binomial Tree: {no_div_bt}")
div_bt = fin.binomial_tree_option_pricing_american_complete(curr_stock_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type, div_amounts, div_times)
print(f"Dividend Binomial Tree: {div_bt}")

# Calculate the Greeks of each
div_times = [datetime(2022, 4, 12)]
print("Call Binomial Trees ------------------------------------")
option_type = 'call'
no_div_greeks = fin.greeks(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type)
div_greeks = fin.greeks_with_dividends(curr_stock_price, strike_price, risk_free_rate, implied_volatility, 0, current_date, options_expiration_date, option_type, div_times, div_amounts)
print(f"No Discrete Dividend Delta: {no_div_greeks[0]}")
print(f"Discrete Dividend Delta: {div_greeks[0]}")

print(f"No Discrete Dividend Gamma: {no_div_greeks[1]}")
print(f"Discrete Dividend Gamma: {div_greeks[1]}")

print(f"No Discrete Dividend Vega: {no_div_greeks[2]}")
print(f"Discrete Dividend Vega: {div_greeks[2]}")

print(f"No Discrete Dividend Theta: {no_div_greeks[3]}")
print(f"Discrete Dividend Theta: {div_greeks[3]}")

print(f"No Discrete Dividend Rho: {no_div_greeks[4]}")
print(f"Discrete DividendRho: {div_greeks[4]}")

print(f"No Discrete Dividend Carry Rho: {no_div_greeks[5]}")
print(f"Discrete Dividend Carry Rho: {div_greeks[5]}")

print("Put Binomial Trees ----------------------------------------")
option_type = 'put'
no_div_greeks = fin.greeks(curr_stock_price, strike_price, risk_free_rate, implied_volatility, continuously_compounding_coupon, current_date, options_expiration_date, option_type)
div_greeks = fin.greeks_with_dividends(curr_stock_price, strike_price, risk_free_rate, implied_volatility, 0, current_date, options_expiration_date, option_type, div_times, div_amounts)
print(f"No Discrete Dividend Delta: {no_div_greeks[0]}")
print(f"Discrete Dividend Delta: {div_greeks[0]}")

print(f"No Discrete Dividend Gamma: {no_div_greeks[1]}")
print(f"Discrete Dividend Gamma: {div_greeks[1]}")

print(f"No Discrete Dividend Vega: {no_div_greeks[2]}")
print(f"Discrete Dividend Vega: {div_greeks[2]}")

print(f"No Discrete Dividend Theta: {no_div_greeks[3]}")
print(f"Discrete Dividend Theta: {div_greeks[3]}")

print(f"No Discrete Dividend Rho: {no_div_greeks[4]}")
print(f"Discrete DividendRho: {div_greeks[4]}")

print(f"No Discrete Dividend Carry Rho: {no_div_greeks[5]}")
print(f"Discrete Dividend Carry Rho: {div_greeks[5]}")