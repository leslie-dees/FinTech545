import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test2.csv")
testout_data = pd.read_csv("testfiles/data/testout_2.1.csv")


# EW Covariance, lambda=0.97
EW_cov = fin.calculate_ewma_covariance_matrix(test_data, 0.97)

are_equal = np.allclose(EW_cov.values, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")