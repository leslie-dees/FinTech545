import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import norm

test_data = pd.read_csv("testfiles/data/test7_1.csv")
testout_data = pd.read_csv("testfiles/data/testout8_1.csv")

print(testout_data)

# VaR form Normal Distribution

fd = fin.fit_normal(test_data)

mu, sigma = fin.mle_normal_distribution_one_input(test_data)

VaR_abs = fin.VaR_error_model(fd.error_model)
VaR_diff_mean = fin.calc_var_normal(0, fd.error_model.std())
print(testout_data)
print(VaR_abs)
print(VaR_diff_mean)
# var_norm = fin.calc_var_normal(mu, sigma, alpha=0.05)


# print(var_norm)