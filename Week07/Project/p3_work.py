import pandas as pd
#import yfinance as yf
import statsmodels.api as sm
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Use the Fama French 3 factor return time series to fit a 4 factor model to the following stocks
# Originally used this section to collect data. Now I have a master dataset that I will just load in:

#  Fama_French_3_Factor_Return = pd.read_csv("Week07/Project/F-F_Research_Data_Factors_daily.csv")
# Fama_French_3_Factor_Return['Date'] = pd.to_datetime(Fama_French_3_Factor_Return['Date'], format='%Y%m%d')
# ten_years_ago = pd.to_datetime('today')-pd.DateOffset(years=10)
# Fama_French_3_Factor_Return = Fama_French_3_Factor_Return[Fama_French_3_Factor_Return['Date'] >= ten_years_ago]

# Carhart_Momentum = pd.read_csv("Week07/Project/F-F_Momentum_Factor_daily.csv")
# Carhart_Momentum['Date'] = pd.to_datetime(Carhart_Momentum['Date'], format='%Y%m%d')
# Carhart_Momentum = Carhart_Momentum[Carhart_Momentum['Date'] >= ten_years_ago]

# # Merge the data into a single dataframe
# merged_data = pd.merge(Fama_French_3_Factor_Return, Carhart_Momentum, on='Date')

# stocks_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ', 'UNH', 'HD', 'PG', 'V', 'BAC', 'MA', 'PFE', 'XOM', 'DIS', 'CSCO']

# # Create a new DataFrame to store stock returns
# stock_returns_data = pd.DataFrame(index=merged_data['Date'])

# # Fetching returns data from yahoo finance because we were not provided with this data.
# for stock_symbol in stocks_list:
#     stock_data = yf.download(stock_symbol, start=ten_years_ago, end=pd.to_datetime('today'))
#     stock_return_col_name = f'{stock_symbol}'
    
#     # Calculate daily returns for the stock
#     stock_returns_data[stock_return_col_name] = stock_data['Adj Close'].pct_change()

# # Merge stock returns into the original merged_data DataFrame
# merged_data = pd.merge(merged_data, stock_returns_data, left_on='Date', right_index=True)
# merged_data.to_csv("Week07/Project/merged_data_with_stock_returns.csv", index=False)

risk_free_rate = 0.0425
merged_data = pd.read_csv("Week07/Project/merged_data_with_stock_returns.csv")
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.set_index('Date', inplace=True)

# Take 10 year average 
mktrf_return = merged_data['Mkt-RF'].mean()
smb_return = merged_data['SMB'].mean()
hml_return = merged_data['HML'].mean()  
mom_return = merged_data['Mom   '].mean()

stocks_list = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'META', 'NVDA', 'BRK-B', 'JPM', 'JNJ', 'UNH', 'HD', 'PG', 'V', 'BAC', 'MA', 'PFE', 'XOM', 'DIS', 'CSCO']

expected_returns = {}
# Based on the past 10 years of factor returns, find the expected annual return of each stock
for stock in stocks_list:

    y = merged_data[stock] - risk_free_rate

    X = merged_data[['Mkt-RF', 'SMB', 'HML', 'Mom   ']]
    X = sm.add_constant(X) # adding constant for intercept

    # Fit a 4 factor model to the stocks
    model = sm.OLS(y, X).fit()
    params = model.params

    epsilon = params['const']
    beta_mktrf = params['Mkt-RF']
    beta_smb = params['SMB']
    beta_hml = params['HML']
    beta_mom = params['Mom   ']

    expected_return = 100*(risk_free_rate + beta_mktrf*mktrf_return + beta_smb*smb_return + beta_hml*hml_return + beta_mom*mom_return + epsilon)
    expected_returns[stock] = expected_return

    print(f"{stock}: {expected_return*100}%")


# Construct an annual covariance matrix for the stocks
# Drop Fama values first
new_df = merged_data.drop(['Mkt-RF', 'SMB', 'HML', 'Mom   ', 'RF'], axis=1)
covar_matrix = fin.calculate_ewma_covariance_matrix(new_df, lambd=0.97)

returns = np.array(list(expected_returns.values()))

# Assume the risk free rate is 0.0425. Find the super efficient portfolio.
er = returns
covar = covar_matrix.to_numpy()

def optimize_risk(R, covar):
    n = len(er)
    w = cp.Variable(n, nonneg=True)
    
    objective = cp.Minimize(cp.quad_form(w, covar))
    constraints = [cp.sum(w) == 1, cp.sum(er @ w) == R]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return {
        'risk': problem.value,
        'weights': w.value,
        'R': R
    }

returns_range = np.arange(0.03, 0.25, 0.005)
optim_portfolios = [optimize_risk(R, covar) for R in returns_range]

# Plot the efficient frontier
plt.plot(np.sqrt([p['risk'] for p in optim_portfolios]), returns_range, label="Efficient Frontier")


# Sharpe Ratios
optim_portfolios_df = pd.DataFrame(optim_portfolios)
optim_portfolios_df['SR'] = (optim_portfolios_df['R'] - 0.03) / np.sqrt(optim_portfolios_df['risk'])
maxSR = optim_portfolios_df['SR'].idxmax()
maxSR_ret = optim_portfolios_df['R'][maxSR]
maxSR_risk = np.sqrt(optim_portfolios_df['risk'][maxSR])

print(f"Portfolio Weights at the Maximum Sharpe Ratio: {optim_portfolios_df['weights'][maxSR]}")
print(f"Portfolio Return : {maxSR_ret}")
print(f"Portfolio Risk   : {maxSR_risk}")
print(f"Portfolio SR     : {optim_portfolios_df['SR'][maxSR]}")

# Plot the efficient frontier and the additional section
plt.plot(np.sqrt([p['risk'] for p in optim_portfolios]), returns_range, label="Efficient Frontier")
plt.scatter(maxSR_risk, maxSR_ret, color='red', label="Max SR Portfolio")

# Additional Section
w = np.arange(0, 2.1, 0.1)
returns_additional = maxSR_ret * w + 0.03 * (1 - w)
risks_additional = maxSR_risk * w
plt.plot(risks_additional, returns_additional, label="Investment A + Rf", color='green')

plt.xlabel("Risk - SD")
plt.ylabel("Portfolio Expected Return")
plt.legend()
plt.show()