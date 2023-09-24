import numpy as np
from numpy.linalg import norm
import pandas as pd
import time

def calculate_ewma_covariance_matrix(df, lambd):
    # Get the number of time steps n and vars m
    n, m = df.shape  
    
    # Initialize the exponentially weighted covariance matrix as a square matrix with dimensions (m, m)
    ewma_cov_matrix = np.zeros((m, m))  
    
    # Calculate the weights and normalized weights for each time step
    # w_{t_i} = (1-lambda)*lambda^{i-1}
    weights = [(1 - lambd) * lambd**(i) for i in range(n)]  
    
    # Calculate the sum of weights to normalize them
    total_weight = sum(weights)  # sum w_{t-j}
    
    # Normalize the weights by dividing each weight by the total weight
    # w_{t_i}^hat = w_{t_i} / sum w_{t-j}
    normalized_weights = [w / total_weight for w in weights]  
    
    # Calculate the means for each variable across all time steps
    means = df.mean()  
    
    # Calculate the exponentially weighted covariance matrix
    for t in range(n):
        # Calculate the deviation of each variable at time t from its mean
        deviation = df.iloc[t, :] - means  
        
        # weighted deviation from means for x and y
        ewma_cov_matrix += normalized_weights[t] * deviation.values.reshape(-1, 1) @ deviation.values.reshape(1, -1)
    ewma_cov_matrix = pd.DataFrame(ewma_cov_matrix)
    return ewma_cov_matrix

def chol_psd(root, a):
    n = a.shape[0]

    # Initialize the root matrix with 0 values
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        s = 0.0

        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal Element
        temp = a[j, j] - s
        if 0.0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. Just set the column to 0 if we have one
        if root[j, j] == 0.0:
            root[j, j+1:] = 0.0
        else:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return root

def is_psd(matrix):
    # Check if a matrix is positive semidefinite
    eigenvalues = np.linalg.eigvals(matrix)
    psd =  np.all(eigenvalues >= 0)
    if psd:
        print("The matrix is positive semidefinite.")
    else:
        print("The matrix is not positive semidefinite.")
        # Print the negative eigenvalues
        negative_eigenvalues = eigenvalues[eigenvalues < 0]
        print("Negative Eigenvalues:", negative_eigenvalues)


def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance
    if np.sum(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(np.dot(invSD, out), invSD)

    # SVD, update the eigenvalue and scale
    vals, vecs = np.linalg.eigh(out)

    vals = np.maximum(vals, epsilon)
   
    T = 1.0 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)

    return out

def higham_near_psd(a, epsilon = 0.0, max_iterations=100):
    # Initialize variables
    delta_S = np.zeros_like(a)  # Initialize Delta S_0 to zero
    X = np.copy(a)             # Initialize Y_0 as a copy of the input matrix a
    Y = np.copy(a)             # Initialize Y_0 as a copy of the input matrix a
    diffY = np.inf

    if not np.all((np.transpose(a) == a)):
        # Check if the input matrix is symmetric; needed for eigenvalue computation
        raise ValueError('Input Matrix is not symmetric')


    iteration = 0
    # Continue iterating until maximum iterations reached or until difference is within tolerance levels
    while iteration < max_iterations:
        iteration += 1
        if diffY < epsilon:
            break

        Yold = np.copy(Y)  # Store the previous Y (Y_k-1)
        R = Y - delta_S    # Calculate R_k = Y_(k-1) - Delta S_(k-1)
        
        # Compute the weighted R (R_k) using a diagonal weight matrix
        W = np.sqrt(np.diag(np.diag(Y)))  # Diagonal weight matrix W

        R_wtd = np.linalg.inv(W) @ (W @ R @ W) @ np.linalg.inv(W) # Apply weight matrix
        
        # Perform the projection onto the space of symmetric positive definite matrices
        d, v = np.linalg.eigh(R_wtd)
        X = v @ np.diag(np.maximum(d, 0)) @ v.T
        
        delta_S = X - R         # Calculate Delta S_k = X_k - R_k

        Y = np.copy(X)
        np.fill_diagonal(Y, 1)  # Y_k = P_U(X_k)
        
        # Compute norms for convergence checking
        diffY = norm(Y - Yold, 'fro') / norm(Y, 'fro') #lambda calc

    return X

# Implement a multivariate normal simulation that allows for simulation directly from a covar matrix or using PCA and parameter for % var explained
def multivariate_normal_simulation(mean, cov_matrix, num_samples, method='Direct', pca_explained_var=None):
    if method == 'Direct':
        cov_matrix = cov_matrix.values
        n = cov_matrix.shape[1]
        # Initialize an array to store the simulation results
        simulations = np.zeros((num_samples, n))
        # Initialize the root matrix
        root = np.zeros((n, n), dtype=np.float64)

        L = chol_psd(root, cov_matrix)

        Z = np.random.randn(n, num_samples)

        # Calculate simulated multivariate normal samples
        for i in range(num_samples):
            simulations[i, :] = mean + np.dot(L, Z[:, i])

        return simulations
    elif method == 'PCA':
        if pca_explained_var is None:
            pca_explained_var = 1.0
        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues in descending order along with eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Normalize eigenvalues to get the proportion of explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        # Determine the number of components needed to explain the desired variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        k = np.argmax(cumulative_variance_ratio >= pca_explained_var) + 1

        # Select the top k eigenvectors and their eigenvalues
        selected_eigenvalues = eigenvalues[:k]
        selected_eigenvectors = eigenvectors[:, :k]

        # Construct a new covariance matrix using the selected eigenvectors and eigenvalues
        new_cov_matrix = np.dot(selected_eigenvectors, np.dot(np.diag(selected_eigenvalues), selected_eigenvectors.T))



        n = cov_matrix.shape[0]
        simulations = np.random.multivariate_normal(mean, new_cov_matrix, num_samples)

        return simulations

def simulate_and_print_norms(cov_matrices, mean_returns, num_samples, cov_matrix_names, method='Direct', pca_explained_var=None):
    for i, (cov_matrix, cov_matrix_name) in enumerate(zip(cov_matrices, cov_matrix_names)):
        # Start timing
        start_time = time.time()

        # Direct Simulation
        simulated_data = multivariate_normal_simulation(mean_returns, cov_matrix, num_samples, method, pca_explained_var)

        # Calculate the covariance matrix of the simulated data
        simulated_covariance = np.cov(simulated_data, rowvar=False)

        # Calculate the Frobenius Norm
        frobenius_norm = np.linalg.norm(cov_matrix - simulated_covariance)

        # End timing
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        if method == 'Direct':
            print("Method: ", method)
        else:
            print(f"Method: {method} Explained Variance: {pca_explained_var}")
        print(f"Simulation {i + 1} - Covariance Matrix: {cov_matrix_name}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Frobenius Norm: {frobenius_norm:.4f}\n")