import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package as fin
import numpy as np

test_data = pd.read_csv("testfiles/data/test5_2.csv")
testout_data = pd.read_csv("testfiles/data/testout_5.5.csv")

psd_data = fin.near_psd(test_data)

# Simulate normal data
num_samples = 100000
cov_matrix = fin.covariance_matrix(psd_data)
mean = 0
simulated_data = fin.multivariate_normal_simulation(mean, cov_matrix, num_samples, method='PCA', pca_explained_var=0.99)

# Calculate covariance matrix
cout = fin.covariance_matrix(simulated_data)

are_equal = np.allclose(cout, fin.covariance_matrix(testout_data), 1e-04, 1e-04)

if are_equal:
    print("Test passed")
else:
    print("Test failed")