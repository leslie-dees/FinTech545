import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')
import fin_package_3 as f3

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Use DailyReturn.csv
dailyreturn = pd.read_csv('Week03/Project/DailyReturn.csv')

# Rename the first column to "Date" and setting to index column
dailyreturn.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
dailyreturn.set_index('Date', inplace=True)

# Calculate the exponentially weighted covariance matrix using your function
v_df = f3.calculate_ewma_covariance_matrix(dailyreturn, 0)

#Compare to non-weighted covariance matrix
p = dailyreturn.cov()
frob = np.linalg.norm(v_df.values - p.values, 'fro')
print(f"Frobenius Norm between unweighted covar matrix and my exp. weighted covar matrix {frob}")

# Vary lambda from 0 to 1
lambda_values = np.linspace(0, 1.0, 16)

lambda_values = lambda_values[:-1]

lambda_values = [0, 0.5, 0.75, 0.9, 0.95, 0.99]
# Use PCA and plot the cumulative variance explained by each eigenvalue for each lambda chosen
explained_variances = []

# Loop through different lambda values
for lambd in lambda_values:
    # Calculate the exponentially weighted covariance matrix using your function
    v = f3.calculate_ewma_covariance_matrix(dailyreturn, lambd)
    v_df = pd.DataFrame(v)
    
    # Calculate the eigenvalues of v_df
    eigenvalues = np.linalg.eigvals(v_df)
    
    # Calculate explained variance ratio using PCA
    pca = PCA()
    pca.fit(v_df)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    explained_variances.append(explained_variance_ratio)

# Plot the cumulative explained variance for each lambda
plt.figure(figsize=(10, 6))
for i, lambd in enumerate(lambda_values):
    plt.plot(range(1, len(explained_variances[i]) + 1), explained_variances[i], label=f'Lambda = {lambd:.2f}')

plt.xlabel('Number of Eigenvalues')
plt.ylabel('Percent Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Eigenvalues')
plt.legend()
plt.grid(True)
plt.savefig('Week03/Project/cum_exp_var_vs_num_eig.png')
plt.show()
# What does this tell us about the values of lambda and the effect it has on the covariance matrix?

