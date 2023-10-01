import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package_3 as f3
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from scipy.optimize import root_scalar

alpha = 0.05
def return_calculate(prices_df, method="DISCRETE", date_column="Date"):
    vars = prices_df.columns
    n_vars = len(vars)
    vars = [var for var in vars if var != date_column]
    
    if n_vars == len(vars):
        raise ValueError(f"date_column: {date_column} not in DataFrame: {vars}")
    
    n_vars = n_vars - 1
    
    p = prices_df[vars].values
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n-1, m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    
    dates = prices_df[date_column].iloc[1:]
    
    # Create a new DataFrame with all columns
    data = {date_column: dates}
    for i in range(n_vars):
        data[vars[i]] = p2[:, i]
    
    out = pd.DataFrame(data)
    
    return out

# Use DailyPrices.csv
prices = pd.read_csv("Week04/DailyPrices.csv")

# Calculate arithmetic returns for all prices
returns = return_calculate(prices)
meta_returns = returns['META']
# Remove the mean from the series so that the mean(META)=0

# Calculate the mean of the "META" column
mean_meta = meta_returns.mean(numeric_only=True)

# Subtract the mean from the "META" column to center the data
centered_returns = meta_returns - mean_meta

print("Mean of META: ", centered_returns.mean(numeric_only=True))

# Calculate VaR:
def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, mean, std_dev)

    return VaR

### Normal Distribution
# calculate std_dv and mean of returns
sigma = centered_returns.std()
mean = centered_returns.mean(numeric_only=True)
norm_var = calc_var_normal(mean, sigma)
print(f"Normal Distribution VaR: {norm_var:.4f}")

### Normal Distribution with exp weighted var l     ambda = 0.94
df = centered_returns.to_frame()
ew_covar = f3.calculate_ewma_covariance_matrix(df, 0.94)
ew_std_dev = np.sqrt(np.diag(ew_covar))
mean = centered_returns.mean(numeric_only=True)
norm_var_exp = calc_var_normal(mean, ew_std_dev)[0]
print(f"Normal Distribution Exp Weighted VaR: {norm_var_exp:.4f}")

### MLE fitted T distribution

# Fixed MLE for t distribution for just the dataframe values input
def mle_t_distribution(y):
    # Define the likelihood function for the t-distribution
    def neg_log_likelihood(params, y):
        mean, var, df = params
        adjusted_y = y - mean
        log_likeli = -np.sum(t.logpdf(adjusted_y, df, loc=mean, scale=var))
        return log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    df_guess = len(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of negative log likelihood
    result = minimize(neg_log_likelihood, initial_params, args=(y,), method='Nelder-Mead')

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    return optimized_mean, optimized_std_dev, optimized_df

# Call the function to estimate parameters and calculate VaR
t_mean, t_std, t_df = mle_t_distribution(centered_returns.values)

t_VaR = t.ppf(alpha, t_df, loc=t_mean, scale=t_std)
print(f"MLE Fitted t-distribution VaR: {t_VaR:.4f}")

### AR(1) method
def simulate_ar1_process(N, alpha, sigma, mu, num_steps):
    # Initialize variables
    y = np.empty((N, num_steps))
    
    for i in range(N):
        # Generate random noise
        epsilon = np.random.normal(0, sigma, num_steps)
        # initial value
        y[i, 0] = mu + epsilon[0]
        
        for t in range(1, num_steps):
            y[i, t] = mu + alpha * (y[i, t - 1] - mu) + epsilon[t]
    
    return y

# Parameters
N = 10000  # Number of simulations
ar_coef = 0.5  # Autoregressive coefficient
sigma = centered_returns.std()  # Standard deviation of centered returns
mu = centered_returns.mean()  # Mean of centered returns
num_steps = len(centered_returns)  # Number of time steps

# Simulate AR(1) process
ar1_simulated_data = simulate_ar1_process(N, ar_coef, sigma, mu, num_steps)

# Extract the last time step values for each simulation run
final_values = ar1_simulated_data[:, -1]

# Calculate VaR by finding the quantile at alpha
var_alpha = np.percentile(final_values, 100 * alpha)

print(f"AR(1) Process VaR: {var_alpha:.4f}")

### Historical Simulation
# Step 1: Calculate Current Portfolio Value
# Assuming you hold 86 shares of META obtained from the portfolio holdings listings
portfolio_value = 86 * prices['META'].iloc[-1]

# Step 2: Simulate N draws from history with replacement
N = 10000
historical_returns = centered_returns.values  # Use centered returns
historical_simulations = np.random.choice(historical_returns, size=(N, len(historical_returns)), replace=True)

# Step 3: Calculate new prices from the returns
simulated_prices = np.zeros_like(historical_simulations)
simulated_prices[:, 0] = prices['META'].iloc[-1]

for i in range(1, len(historical_returns)):
    simulated_prices[:, i] = simulated_prices[:, i - 1] * (1 + historical_simulations[:, i])

# Step 4: Price each asset for all N draws
# (This step is implicitly done when we calculate portfolio values)

# Step 5: Calculate portfolio Value for each N Draw
portfolio_values = simulated_prices[:, -1]

# Step 6: Get the alpha percentile of the distribution
alpha_percentile = np.percentile(portfolio_values, 100*alpha)
VaR = -alpha_percentile/portfolio_value
print(f"Historical Simulation VaR: {VaR:.4f}")