import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import norm

test_data = pd.read_csv("testfiles/data/test7_1.csv")
testout_data = pd.read_csv("testfiles/data/testout8_4.csv")

# ES from Normal Distribution
fd = fin.fit_normal(test_data)

print(testout_data)

ES_abs = fin.ES_error_model(fd.error_model)
print(ES_abs)
ES_diff_norm = fin.ES_error_model(norm(0, fd.error_model.std()))
print(ES_diff_norm)
# print(ES_abs)
# print(ES_diff_mean)