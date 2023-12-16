import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import t

test_data = pd.read_csv("testfiles/data/test7_2.csv")
testout_data = pd.read_csv("testfiles/data/testout8_2.csv")



# VaR form Normal Distribution

mu, sigma, nu, fitted_model = fin.fit_general_t(test_data)

VaR_abs = fin.VaR_error_model(fitted_model.error_model)
VaR_diff_mean = fin.VaR_error_model(t(df=nu, loc=0, scale=sigma))

print(testout_data)

print(VaR_abs)
print(VaR_diff_mean)