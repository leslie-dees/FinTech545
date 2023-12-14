import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test1.csv")
testout_data = pd.read_csv("testfiles/data/testout_1.2.csv")

# Correlation Missing data, skip missing rows

corr_matrix = fin.correlation_matrix(test_data)

are_equal = np.allclose(corr_matrix.values, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")