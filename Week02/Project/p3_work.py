import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')

import fin_package
import numpy as np

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

fin_package.plot_acf_pacf(y, plot_type = "AR")