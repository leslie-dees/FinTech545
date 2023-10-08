import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import pandas as pd
from tqdm import tqdm
from scipy.stats import t

# Using Portfolio.csv & DailyPrices.csv, assume expected return on all stocks is 0
portfolio = pd.read_csv("Week05/Project/portfolio.csv")
dailyprices = pd.read_csv("Week05/DailyPrices.csv")

returns = fin.return_calculate(dailyprices).drop("Date", axis=1)
# Portfolios of A, B, C
portfolio_a = portfolio[portfolio['Portfolio'] == "A"]
portfolio_b = portfolio[portfolio['Portfolio'] == "B"]
portfolio_c = portfolio[portfolio['Portfolio'] == "C"]
# Fit a Generalized T model to each stock
t_dist_dict = {}
stocks = portfolio['Stock']
for stock in tqdm(stocks, desc="Fitting T-Distribution to Stocks"):
    s_df, s_mean, s_std = t.fit(dailyprices[stock])
    stock_dict = {
        'mean': s_mean,
        'std_dev': s_std,
        'df': s_df
    }
    t_dist_dict[stock] = stock_dict

lambd = 0.97
# Calculate VaR of A
print("Value at Risk: Portfolio A")
A_var = -fin.calculate_portfolio_var(portfolio_a, dailyprices, returns, lambd)

# Calculate VaR of B
print("Value at Risk: Portfolio B")
B_var = -fin.calculate_portfolio_var(portfolio_b, dailyprices, returns, lambd)

# Calculate VaR of C
print("Value at Risk: Portfolio C")
C_var = -fin.calculate_portfolio_var(portfolio_c, dailyprices, returns, lambd)

# Calculate VaR of Total Portfolio
print("Value at Risk: Portfolio Whole")
Port_var = -fin.calculate_portfolio_var(portfolio, dailyprices, returns, lambd)

# Calculate ES of A
es_vals_a = fin.portfolio_es(portfolio_a, t_dist_dict, dist = "T")
print(f"Expected Shortfall: Portfolio A -- {es_vals_a}")

# Calculate ES of B
es_vals_b = fin.portfolio_es(portfolio_b, t_dist_dict, dist = "T")
print(f"Expected Shortfall: Portfolio B -- {es_vals_b}")
# Calculate ES of C
es_vals_c = fin.portfolio_es(portfolio_c, t_dist_dict, dist = "T")
print(f"Expected Shortfall: Portfolio C -- {es_vals_c}")
# Calculate ES of Total Portfolio
es_vals_all = fin.portfolio_es(portfolio, t_dist_dict, dist = "T")
print(f"Expected Shortfall: Portfolio All -- {es_vals_all}")