import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test1.csv")
testout_data = pd.read_csv("testfiles/data/testout_1.4.csv")

# Covariance Missing data, Pairwise
corr_matrix = fin.correlation_matrix(test_data, False)

are_equal = np.allclose(corr_matrix.values, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")