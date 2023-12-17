import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

ff3 = pd.read_csv("Week08/F-F_Research_Data_Factors_daily.CSV")
mom = pd.read_csv("Week08/F-F_Momentum_Factor_daily.CSV")
returns = pd.read_csv("Week08/DailyReturn.csv")

# Join the FF3 data with the Momentum Data
ffData = pd.merge(ff3, mom, on='Date', how='inner')
ffData.rename(columns={'Mkt-RF': 'Mkt_RF'}, inplace=True)
ffData.iloc[:, 1:] /= 100  # Convert percentage values to decimal
ffData['Date'] = pd.to_datetime(ffData['Date'], format="%Y%m%d")

returns['Date'] = pd.to_datetime(returns['Date'], format="%m/%d/%Y")

# Join the FF3+1 to Stock data - filter to stocks we want
stocks = ['AAPL', 'MSFT', 'BRK-B', 'CSCO', 'JNJ']
to_reg = pd.merge(returns[['Date', 'SPY'] + stocks], ffData, on='Date', how='inner')

xnames = ['Mkt_RF', 'SMB', 'HML', 'Mom   ']

# OLS Regression for all Stocks
X = np.column_stack([np.ones(to_reg.shape[0]), to_reg[xnames].values])

# Regress against y matrix
Y = to_reg[stocks].values
betas = np.linalg.inv(X.T @ X) @ X.T @ Y
betas = betas.T[:, 1:X.shape[1]]  # matrix of our Betas

# Max and min dates
max_dt = to_reg['Date'].max()
min_dt = max_dt - pd.DateOffset(years=10)
to_mean = ffData[(ffData['Date'] >= min_dt) & (ffData['Date'] <= max_dt)].copy()

# Historic daily factor returns
exp_Factor_Return = to_mean[xnames].mean()
expFactorReturns = pd.DataFrame({'Factor': xnames, 'Er': exp_Factor_Return})

# Scale returns and covar
stockMeans = np.log(1.0 + betas @ exp_Factor_Return) * 255
covar = np.cov(np.log(1.0 + Y).T) * 255

# Sharpe ratio function
def sr(w):
    _w = np.array(w)
    m = _w @ stockMeans - 0.0025
    s = np.sqrt(_w @ covar @ _w)
    return m / s

n = len(stocks)

def objective_function(w):
    w = np.array(w)
    m = w @ stockMeans - 0.0025
    s = np.sqrt(w @ covar @ w)
    return -m / s

constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

initial_weights = np.ones(n) / n

# Weigh tbounds
bounds = tuple((0, None) for _ in range(n))

result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraint)

# Get the optimized weights
optimal_weights = result.x / np.sum(result.x)

# weight = portfolio weight, cEr = expected return for each stock
opt_weights_df = pd.DataFrame({'Stock': stocks, 'Weight': optimal_weights, 'cEr': stockMeans * optimal_weights})

print("Max SR Portfolio")
print(opt_weights_df)

updated = pd.read_csv("Week08/updated_prices.csv")
updated['Date'] = pd.to_datetime(updated['Date'], format="%m/%d/%Y")
upReturns = fin.return_calculate(updated)

n = upReturns.shape[0]
m = len(stocks)

pReturn = np.zeros(n)
weights = np.zeros((n, len(optimal_weights)))
lastW = optimal_weights.copy()
matReturns = upReturns[stocks].to_numpy()

for i in range(n):
    weights[i, :] = lastW

    # Update weights by return
    lastW = lastW * (1.0 + matReturns[i, :])
    
    # Portfolio return is the sum of the updated weights
    pR = np.sum(lastW)
    
    # Normalize the weights
    lastW = lastW / pR
    
    pReturn[i] = pR - 1

# Set the portfolio return in the Update Return DataFrame
upReturns['Portfolio'] = pReturn

# Calculate the total return
total_ret = np.exp(np.sum(np.log(pReturn + 1))) - 1

# Calculate the Carino K
k = np.log(total_ret + 1) / total_ret

carino_k = np.log(1.0 + pReturn) / pReturn / k

# Calculate the return attribution
first = matReturns * weights
second = np.multiply(first, np.expand_dims(carino_k, axis=0).T)
attrib = pd.DataFrame(second, columns=stocks)

# Set up a DataFrame for output
attributions = pd.DataFrame(index=["TotalReturn", "ReturnAttribution"])

for s in stocks + ['Portfolio']:
    # Total Stock return over the period
    tr = np.exp(np.sum(np.log(upReturns[s] + 1))) - 1
    
    # Attribution Return (total portfolio return if we are updating the portfolio column)
    atr = np.sum(attrib[s]) if s != 'Portfolio' else tr
    
    # Set the values
    attributions[s] = [tr, atr]

# Check that the attribution sums back to the total Portfolio return
is_sum_close = np.isclose(np.sum(attributions.loc["ReturnAttribution", stocks].values), total_ret)

print(attributions)
print("Attribution sums to total Portfolio return:", is_sum_close)

# Realized Volatility Attribution (Risk attribution)

# Y is our stock returns scaled by their weight at each time
Y = matReturns * weights

# Set up X with the Portfolio Return
X = np.column_stack([np.ones(n), pReturn])

B = np.linalg.inv(X.T @ X) @ X.T @ Y
B = B[1,:]

# Component standard deviation
cSD = B * np.std(pReturn)

# Check that the sum of component SD is equal to the portfolio SD
is_sum_cSD_close = np.isclose(np.sum(cSD), np.std(pReturn))

# Add the Vol attribution to the output
vol_attribution = pd.DataFrame(
    {"Value": ["VolAttribution"], **{s: [cSD[i]] for i, s in enumerate(stocks)}, "Portfolio": [np.std(pReturn)]}
)

# Concatenate the new attribution to the existing DataFrame
attributions = pd.concat([attributions, vol_attribution], ignore_index=True, sort=False)
attributions = attributions.drop('Value', axis=1)

attributions.index = ["Total Return", "Return Attribution", "Volatility Attribution"]

print(attributions)