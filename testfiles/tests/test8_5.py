import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import t

test_data = pd.read_csv("testfiles/data/test7_1.csv")
testout_data = pd.read_csv("testfiles/data/testout8_4.csv")

# ES from Normal Distribution
mu, sigma, nu, error_model = fin.fit_general_t(test_data)

print(testout_data)

ES_abs = fin.calc_expected_shortfall_t(0, sigma, nu, alpha=0.05)
print(ES_abs)
#ES_diff_norm = fin.ES_error_model(t(df = nu, loc=0, scale=sigma))
# print(ES_diff_norm)
# print(ES_abs)
# print(ES_diff_mean)