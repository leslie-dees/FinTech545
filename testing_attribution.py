import numpy as np
import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin

# Set seed for reproducibility
np.random.seed(42)

# Generate weights
weights_df = pd.read_csv('Problem4_startWeigth.csv')
returns_df = pd.read_csv('Problem4_returns.csv')


corr = fin.correlation_matrix(returns_df)

std_devs = returns_df.std()

weights = weights_df.values
weights = weights.flatten()

diag_std = np.diag(std_devs)

covar = diag_std @ corr @ diag_std

# True Portfolio Standard Deviation
true_portfolio_std = np.sqrt(weights @ covar @ weights)

# True Portfolio Volatility Attribution
volatility_attribution = weights * (covar @ weights) / true_portfolio_std

# Display True Portfolio Standard Deviation and Volatility Attribution
print("True Portfolio Standard Deviation:", true_portfolio_std)
print("\nTrue Portfolio Volatility Attribution:")
for i, attribution in enumerate(volatility_attribution, start=1):
    print(f"Asset {i} Vol Contribution: {attribution}")

    # Test if attribution sums correctly
sums_correctly = np.isclose(np.sum(volatility_attribution), true_portfolio_std)

# Display the result
print("Test if attribution sums correctly:", sums_correctly)

# Simulate 1000 Asset Returns
n = 1000
mean_returns = np.zeros(len(std_devs))
asset_returns_simulation = np.random.multivariate_normal(mean=mean_returns, cov=covar, size=n)

# Portfolio Returns
portfolio_returns = asset_returns_simulation @ weights

# Display Portfolio Volatility and Actual Portfolio Volatility
portfolio_volatility = np.std(portfolio_returns)
print("Portfolio Volatility:", portfolio_volatility)
print("Actual Portfolio Volatility:", true_portfolio_std)

# Volatility Attribution using OLS Regression
X = np.column_stack([np.ones(n), portfolio_returns])
Y = asset_returns_simulation * weights

# OLS Regression to get Beta values
B_s_p = np.linalg.lstsq(X, Y, rcond=None)[0][1, :]

# Calculate volatility contribution
volatility_contribution = B_s_p * np.std(portfolio_returns)

# Display Volatility Attribution
print("\nVolatility Attribution:")
for i, contribution in enumerate(volatility_contribution, start=1):
    print(f"Asset {i} Vol Contribution: {contribution}")

# Test if attribution sums correctly
sums_correctly_ols = np.isclose(np.sum(volatility_contribution), np.std(portfolio_returns))
print("Test if attribution sums correctly:", sums_correctly_ols)
