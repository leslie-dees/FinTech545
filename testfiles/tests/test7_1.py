import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test7_1.csv")
testout_data = pd.read_csv("testfiles/data/testout7_1.csv")

testout_mu = testout_data['mu'].values
testout_sigma = testout_data['sigma'].values

mu, sigma = fin.mle_normal_distribution_one_input(test_data)

mu_equal = np.allclose(testout_mu, mu, 1e-03, 1e-03)
sigma_equal = np.allclose(testout_sigma, sigma, 1e-03, 1e-03)

if mu_equal and sigma_equal:
    print("Test passed")
else:
    print("Test failed")