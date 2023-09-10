import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')

import fin_package
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Order of the MA process
N = 1
# Number of time steps
num_steps = 1000
# Burn-in period
burn_in = 50

# Generate normal white noise
mean = 0.0
std_dev = 1.0
np.random.seed(0)
e = np.random.normal(mean, std_dev, num_steps + burn_in)

# Simulate the MA(N) process
#y, mean_y, var_y = fin_package.simulate_MA(N, num_steps, e, burn_in, mean, plot_y=True)


# Simulate the AR(N) process
y, mean_y, var_y = fin_package.simulate_AR(N, num_steps, e, burn_in, mean, plot_y=True)

# Set custom styling for the plots
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'red'
plt.rcParams['ytick.color'] = 'red'
plt.rcParams['text.color'] = 'white'

# Plot the ACF and PACF with red lines
plt.figure(figsize=(12, 6))

# ACF plot
ax1 = plt.subplot(121)
plot_acf(y, lags=40, ax=ax1, color='red')
plt.title("Autocorrelation Function (ACF)")

# PACF plot
ax2 = plt.subplot(122)
plot_pacf(y, lags=40, ax=ax2, color='red')
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()