import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test5_1.csv")
testout_data = pd.read_csv("testfiles/data/testout_5.1.csv")

print(testout_data)

# chol_psd

# are_equal = np.allclose(chol_psd_matrix, testout_data.values)

# if are_equal:
#     print("Test passed")
# else:
#     print("Test failed")