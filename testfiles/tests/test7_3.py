import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import t, kurtosis
from scipy.optimize import minimize

test_data = pd.read_csv("testfiles/data/test7_3.csv")
testout_data = pd.read_csv("testfiles/data/testout7_3.csv")

# get data
df_y = test_data[['y']]
df_x = test_data.drop(columns=['y'])

# Fit regression model with T errors
def fit_regression_t(x, y):
    n = x.shape[0]
    nB = x.shape[1]

    __x = np.column_stack((np.ones(n), x))
    __y = y

    # Fit a general T distribution given an x input
    b_start = np.linalg.inv(__x.T @ __x) @ __x.T @ __y
    e = __y - __x @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / kurtosis(e) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)
   
    def _gtl(params):
        mu, s, nu, *beta = params

        xm = __y.values.reshape(-1, 1) - (__x @ beta).reshape(-1, 1)
        new_params = [mu, s, nu]
        return fin.general_t_ll(new_params, xm)

    # Initial parameter values
    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))

    # Optimization using Nelder Mead
    result = minimize(_gtl, initial_params, method='Nelder-Mead')

    m, s, nu, *beta = result.x

    return m, s, nu, *beta

mu, sigma, nu, alpha, *beta = fit_regression_t(df_x, df_y)
print(mu)
print(sigma)
print(nu)
print(alpha)
print(beta)

print(testout_data)