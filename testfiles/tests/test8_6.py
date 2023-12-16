import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np
from scipy.stats import norm

test_data = pd.read_csv("testfiles/data/test7_2.csv")
testout_data = pd.read_csv("testfiles/data/testout8_6.csv")

mu, sigma, nu, fitted_model = fin.fit_general_t(test_data)
sim = fitted_model.evaluate(np.random.rand(100000))

print(testout_data)

print(fin.ES_simulation(sim))
print(fin.ES_simulation(sim - np.mean(sim)))