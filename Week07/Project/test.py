import numpy as np
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

def binomial_tree_option_pricing_american_complete(underlying_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type, div_amounts = None, div_times = None):
    S = underlying_price
    X = strike_price
    T = ttm
    r = risk_free_rate
    sigma = implied_volatility
    N = num_steps

    if (div_amounts is None) or (div_times is None) or len(div_amounts) == 0 or len(div_times) == 0 or div_times[0] > N:
        return fin.binomial_tree_option_pricing_american(S, X, T, risk_free_rate, risk_free_rate, implied_volatility, num_steps, option_type)
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    if option_type == 'call':
        z = 1
    else:
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i + 1

    nDiv = len(div_times)
    n_nodes = nNodeFunc(N)
    option_values = np.empty(n_nodes + 1)  # Increase the size by 1

    for j in range(div_times[0], -1, -1):  # Use a float range for j
        for i in range(j, -1, -1):  # Use a float range for i
            idx = idxFunc(i, j)
            price = S * (u ** i) * (d ** (j - i))

            if j < div_times[0]:
                # times before or at the dividend working backward induction
                option_values[idx] = max(0, z * (price - X))
                option_values[idx] = max(option_values[idx], df * (pu * option_values[idxFunc(i + 1, j + 1)] + pd * option_values[idxFunc(i, j + 1)]))
            else:
                # time after the dividend
                val_no_exercise = binomial_tree_option_pricing_american_complete(price-div_amounts[0], X, ttm-div_times[0]*dt, risk_free_rate, implied_volatility, N-div_times[0], option_type, div_amounts[1:nDiv], div_times[1:nDiv] - div_times[0])
                val_exercise = max(0, z * (price - X))
                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[1]


# p = binomial_tree_option_pricing_american_complete(underlying_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type, div_amounts = None, div_times = None)

# 8.267777659056069
# 9.116785549216589
div_amt = np.array([1])
div_times = np.array([1])

two = fin.binomial_tree_option_pricing_american_complete(100, 100, 0.5, 0.08, 0.3, 2, 'call', div_amt, div_times)
print(two)