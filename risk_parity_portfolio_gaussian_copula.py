import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
from scipy.stats import spearmanr, norm

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
#stocks = ['MSFT', 'BRK-B', 'CSCO', 'JNJ']

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

# Function for Portfolio Volatility
def pvol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function for Component Standard Deviation
def pcsd(weights, cov_matrix):
    pvol_value = pvol(weights, cov_matrix)
    csd = weights * (cov_matrix @ weights) / pvol_value
    return csd

# Sum Square Error of cSD
def sse_csd(weights, cov_matrix):
    csd = pcsd(weights, cov_matrix)
    mcsd = np.sum(csd) / len(weights)
    dcsd = csd - mcsd
    se = dcsd * dcsd
    return 1.0e5 * np.sum(se) # Add a large multiplier for better convergence

n = len(stocks)

# Initial weights with boundary at 0
initial_weights = np.ones(n) / n

# Define the objective function
def objective_function(weights):
    return sse_csd(weights, covar)

# Equality constraint: sum of weights = 1
constraint = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Optimization using scipy
result = minimize(objective_function, initial_weights, constraints=constraint, method='SLSQP')

# Extract optimized weights
optimized_weights = result.x

# Calculate portfolio statistics with optimized weights
portfolio_volatility_optimized = pvol(optimized_weights, covar)
component_std_dev_optimized = pcsd(optimized_weights, covar)
sum_square_error_csd_optimized = sse_csd(optimized_weights, covar)

# Create RPWeights DataFrame with optimized weights
RPWeights_optimized = pd.DataFrame({
    'Stock': [str(stock) for stock in stocks],
    'Weight': optimized_weights,
    'cEr': stockMeans * optimized_weights,
    'CSD': pcsd(optimized_weights, covar)
})

print("RPWeights (Optimized):")
print(RPWeights_optimized)

# RP on Simulated ES

# Remove the mean
mean_returns = np.mean(Y, axis=0)
Y = Y - mean_returns

# Fit T Models to the returns
n = Y.shape[1]
m = Y.shape[0]
models = []
U = np.zeros((m, n))

for i in range(n):
    if stocks[i] == 'AAPL':
        model_i = fin.fit_normal(Y[:, i])
    else:
        m, s, nu, model_i = fin.fit_general_t(Y[:, i])
    models.append(model_i)
    U[:, i] = model_i.u

nSim = 5000

# Gaussian Copula - Calculate Spearman correlation matrix using spearmanr
# AAPL would not work, so removed it because would not fit a general t...
corsp, _ = spearmanr(U, axis=0)

# Simulate uniform random variables using the copula
simU = norm.cdf(np.random.multivariate_normal(np.zeros(n), corsp, size=nSim)).T

simReturn = np.zeros_like(simU)

# Evaluate simulated returns using T distribution models
for i in range(n):
    simReturn[:, i] = models[i].evaluate(simU[:, i])

# Internal ES function
# Internal ES function

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

def ES(a, alpha=0.05):
    x = np.sort(a)

    nup = int(my_ceil(n * alpha))
    ndn = int(my_floor(n *alpha))


    v = 0.5 * (x[nup] + x[ndn])

    es = np.mean(x[x <= v])
    return -es

def _ES(weights):
    weights = np.array(weights)
    portfolio_returns = simReturn.T @ weights
    es_value = ES(portfolio_returns)

    return es_value

# Function for the component ES
def CES(weights):
    weights = np.array(weights)
    n = len(weights)
    ces = np.zeros(n)
    es_value = _ES(weights)
    epsilon = 1e-6
    
    for i in range(n):
        old_weight = weights[i]
        weights_copy = weights.copy()
        weights_copy[i] += epsilon
        new_es = _ES(weights_copy)
        ces[i] = old_weight * (new_es - es_value) / epsilon
    
    return ces

# SSE of the Component ES
def SSE_CES(weights):
    ces_values = CES(weights)
    mean_ces = np.mean(ces_values)
    ces_values -= mean_ces
    return 1e3 * ces_values.T @ ces_values

# Optimize to find RP based on Expected Shortfall
n = len(stocks)

# Initial weights with boundary at 0
initial_weights_es = np.ones(n) / n

# Define the objective function for ES
def objective_function_es(weights):
    return SSE_CES(weights)

# Equality constraint: sum of weights = 1
constraint_es = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Optimization using scipy for ES
result_es = minimize(objective_function_es, initial_weights_es, constraints=constraint_es)

# Extract optimized weights for ES
optimized_weights_es = result_es.x

# Calculate portfolio statistics with optimized weights for ES
portfolio_ces_optimized = _ES(optimized_weights_es)
component_ces_optimized = CES(optimized_weights_es)

# Create ES_RPWeights DataFrame with optimized weights
ES_RPWeights_optimized = pd.DataFrame({
    'Stock': [str(stock) for stock in stocks],
    'Weight': optimized_weights_es,
    'cEr': stockMeans * optimized_weights_es,
    'CES': component_ces_optimized
})

print("ES_RPWeights (Optimized):")
print(ES_RPWeights_optimized)