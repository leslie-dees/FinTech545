import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np



test_data = pd.read_csv("testfiles/data/testout_1.3.csv")
testout_data = pd.read_csv("testfiles/data/testout_3.3.csv")

# near_psd covariance

near_psd_matrix = fin.higham_nearestPSD(test_data)

print(test_data)

print(near_psd_matrix)

print(testout_data)

are_equal = np.allclose(near_psd_matrix, testout_data.values)

if are_equal:
    print("Test passed")
else:
    print("Test failed")