import pandas as pd
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')

import fin_package_3 as f3
import numpy as np
import time
from sklearn.decomposition import PCA

# Use DailyReturn.csv
dailyreturn = pd.read_csv('Week03/Project/DailyReturn.csv')

# Rename the first column to "Date" and setting to index column
dailyreturn.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
dailyreturn.set_index('Date', inplace=True)

mean_returns = dailyreturn.mean().values
cov_matrix = dailyreturn.cov()


# Generate a corr matrix and var vector 2 ways:
# # Standard Pearson corr/var

pearson_corr_matrix = dailyreturn.corr()
pearson_var_vector = dailyreturn.var()
pearson_var_vector = pearson_var_vector[:, np.newaxis]

# # Exponentially weighted lambda = 0.97
exp_weight_cov_matrix = f3.calculate_ewma_covariance_matrix(dailyreturn, 0.97)
exp_weight_var_vector = exp_weight_cov_matrix.values.diagonal()
exp_weight_var_vector = exp_weight_var_vector[:, np.newaxis]
exp_weight_corr_matrix = exp_weight_cov_matrix / np.outer(np.sqrt(exp_weight_var_vector), np.sqrt(exp_weight_var_vector))

# Combine these to form 4 different covar matricies
# Pearson corr + var
pearson_cov_matrix = pearson_corr_matrix + pearson_var_vector
# Pearson corr + EW var
pearson_ew_var_matrix = pearson_corr_matrix + exp_weight_var_vector
# Exp weighted corr + var
exp_weight_cov_matrix = exp_weight_corr_matrix + pearson_var_vector
# Exp weighted corr + EW var
exp_weight_ew_var_matrix = exp_weight_corr_matrix + exp_weight_var_vector

# Simulate 25,000 draws:
num_samples = 25000

# # Direct Simulation

# Example usage:
# Define a list of covariance matrices and their corresponding names
cov_matrices = [pearson_cov_matrix, pearson_ew_var_matrix, exp_weight_cov_matrix, exp_weight_ew_var_matrix]
cov_matrix_names = ["Pearson + var", "Pearson + EW var", "Exp weighted + var", "Exp weighted + EW var"]

# Simulate and print norms for each covariance matrix in the list
f3.simulate_and_print_norms(cov_matrices, mean_returns, num_samples, cov_matrix_names, method='Direct')

# # PCA with 100% Explained
f3.simulate_and_print_norms(cov_matrices, mean_returns, num_samples, cov_matrix_names, method='PCA', pca_explained_var=1.0)

# PCA wih 75% Explained
f3.simulate_and_print_norms(cov_matrices, mean_returns, num_samples, cov_matrix_names, method='PCA', pca_explained_var=0.75)

# PCA with 50% Explained
f3.simulate_and_print_norms(cov_matrices, mean_returns, num_samples, cov_matrix_names, method='PCA', pca_explained_var=0.5)