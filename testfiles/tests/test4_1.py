import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/testout_3.1.csv")
testout_data = pd.read_csv("testfiles/data/testout_4.1.csv")

# chol_psd
chol_psd_matrix = fin.chol_psd(test_data)

are_equal = np.allclose(chol_psd_matrix, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")