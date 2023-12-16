import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import norm

test_data = pd.read_csv("testfiles/data/test7_2.csv")
testout_data = pd.read_csv("testfiles/data/testout8_3.csv")

"""
8.2 test values:
   VaR Absolute  VaR Diff from Mean
   0.04153       0.08747
Mine:
VaR Absolute:           0.04152951819647109
VaR Diff from Mean:     0.0874699240882966
"""

# VaR from Simulation -- compare to 8.2 values

mu, sigma, nu, fitted_model = fin.fit_general_t(test_data)

sim = fitted_model.evaluate(np.random.rand(100000))
print(testout_data)

print(fin.VaR_simulation(sim))
print(fin.VaR_simulation(sim - np.mean(sim)))