import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test1.csv")
testout_data = pd.read_csv("testfiles/data/testout_1.1.csv")


# Covariance Missing data, skip missing rows

cov_matrix = fin.covariance_matrix(test_data)

are_equal = np.allclose(cov_matrix.values, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")