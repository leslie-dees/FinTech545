import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test2.csv")
testout_data = pd.read_csv("testfiles/data/testout_2.2.csv")

# Covariance with EW Variance (l=0.94), EW Correlation (l=0.97)

EW_corr = fin.calculate_ewma_correlation_matrix(test_data, 0.97, 0.94)

are_equal = np.allclose(EW_corr.values, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")