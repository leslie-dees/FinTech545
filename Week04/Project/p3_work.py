import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package_3 as f3
from scipy.stats import norm

def calculate_portfolio_var(portfolio, price_df, returns_df, lambd, alpha = 0.05):
    # calculate total portfolio value
    portfolio_value = 0.0
    # create array to store each stock's value
    delta = []
    for _, row in portfolio.iterrows():
        stock_value = row['Holding']*price_df[row['Stock']].iloc[-1]
        portfolio_value += stock_value
        delta.append(stock_value)

    print(f"Portfolio Value: {portfolio_value}")
    delta = np.array(delta)
    normalized_delta = delta / portfolio_value
    
    exp_weighted_cov = f3.calculate_ewma_covariance_matrix(returns, lambd)
    exp_weighted_std = np.sqrt(np.diagonal(exp_weighted_cov))
    
    # Create a dictionary to store column titles and corresponding exp_weighted_std values
    result_dict = {column: std for column, std in zip(returns_df.columns, exp_weighted_std)}
    
    exp_weighted_std_portfolio = np.array([result_dict[stock] for stock in portfolio['Stock']])

    p_sig = np.sqrt(np.dot(np.dot(normalized_delta, exp_weighted_std_portfolio), normalized_delta))
    
    VaR = -delta * norm.ppf(1-alpha)*p_sig
    total_VaR = sum(VaR)

    print(f"Porftolio Value at Risk: ${total_VaR}\n")
    return total_VaR

# Using Portfolio and DailyPrices assume the expected return on all stocks is 0
portfolio = pd.read_csv("Week04/Project/portfolio.csv")
port_a = portfolio[portfolio['Portfolio'] == "A"]
port_b = portfolio[portfolio['Portfolio'] == "B"]
port_c = portfolio[portfolio['Portfolio'] == "C"]

# Load in Prices and Returns
prices = pd.read_csv("Week04/DailyPrices.csv")
returns = pd.read_csv("Week04/DailyReturn.csv").drop('Date', axis=1)


# Using exp weighted covar lambda = 0.94, calculate VaR of each port (VaR as $)
print("Calculating individual portfolio VaR, Lambda = 0.94")
calculate_portfolio_var(port_a, prices, returns, lambd = 0.94)
calculate_portfolio_var(port_b, prices, returns, lambd = 0.94)
calculate_portfolio_var(port_c, prices, returns, lambd = 0.94)

# Using exp weighted covar lambda = 0.94, calculate VaR of total holdings (VaR as $)
print("Calculating total portfolio VaR, Lambda = 0.94")
calculate_portfolio_var(portfolio, prices, returns, lambd = 0.94)

# Using exp weighted covar lambda = test_lambd, calculate VaR of each port (VaR as $)
test_lambd = 0.80
print(f"Calculating individual portfolio VaR, Lambda = {test_lambd}")
calculate_portfolio_var(port_a, prices, returns, lambd = test_lambd)
calculate_portfolio_var(port_b, prices, returns, lambd = test_lambd)
calculate_portfolio_var(port_c, prices, returns, lambd = test_lambd)

# Using exp weighted covar lambda = test_lambd, calculate VaR of total holdings (VaR as $)
print(f"Calculating total portfolio VaR, Lambda = {test_lambd}")
calculate_portfolio_var(portfolio, prices, returns, lambd = test_lambd)