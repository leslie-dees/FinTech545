from scipy.stats import moment
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, lognorm
from scipy.optimize import minimize
from scipy.integrate import quad
import pandas as pd
import time
from tqdm import tqdm
import math
from datetime import datetime

def covariance_matrix(input_df, skipna=True):
    # calculate the covariance matrix either pairwise or skipping rows

    if skipna:
        return input_df.dropna().cov()
    else:
        return input_df.cov()

def correlation_matrix(input_df, skipna=True):
    # calculate the correlation matrix either pairwise or skipping rows
    if skipna:
        return input_df.dropna().corr()
    else:
        return input_df.corr()

def first4Moments(sample, excess_kurtosis=True):
    # Calculate the raw moments
    mean_hat = moment(sample, moment=1)
    var_hat = moment(sample, moment=2, nan_policy='omit')

    # Calculate skewness and kurtosis without dividing
    skew_hat = moment(sample, moment=3)
    kurt_hat = moment(sample, moment=4)

    # Calculate excess kurtosis if excess_kurtosis is True, otherwise return regular kurtosis
    if excess_kurtosis:
        excessKurt_hat = kurt_hat - 3  # Excess kurtosis
        return mean_hat, var_hat, skew_hat, excessKurt_hat
    else:
        return mean_hat, var_hat, skew_hat, kurt_hat  # Regular kurtosis
    
def calc_estimated_kurtosis(sample_size, num_samples, mean, std_dev):
    # Array to keep kurtosis values for 
    estimated_kurtosis_vals = []
    estimated_skew_vals = []

    # Test a new kurtosis for the number of sample distributions available
    for _ in tqdm(range(num_samples), desc="Generating Samples"):
        # Create random normal sample distribution
        sample = np.random.normal(mean, std_dev, sample_size)

        # Calculate kurtosis using your function (first4Moments)
        _, _, skew, kurtosis = first4Moments(sample, excess_kurtosis=False)

        estimated_kurtosis_vals.append(kurtosis)
        estimated_skew_vals.append(skew)

    # Average the estimated kurtosis using your function      
    averaged_estimated_kurtosis = np.mean(estimated_kurtosis_vals)
    averaged_estimated_skew = np.mean(estimated_skew_vals)
    return averaged_estimated_kurtosis, estimated_kurtosis_vals, averaged_estimated_skew, estimated_skew_vals

def perform_ols(X, y, visualize_error=False):
    # Add a constant term to X matrix for the intercept
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    # Calculate error vector
    error_vector = model.resid
    
    # visualize error if desired
    if visualize_error:
        # Visualize the error distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(error_vector, kde=True, color='blue', bins=100)
        plt.title("Error Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()
    
    print("Error Vector: ", error_vector)
    averaged_error_vector = np.mean(error_vector)
    print("Averaged Error Vector: ", averaged_error_vector)
    variance_error_vector = np.var(error_vector)
    print("Variance Error Vector: ", variance_error_vector)
    return error_vector

def mle_normal_distribution(X, y, perform_hypothesis_test = False):
    # Define the likelihood function for the normal distribution
    def log_likelihood(mean, var, X):
        # Number of values in X
        n = len(X)
        # Adjust x to be centered around the mean
        adjusted_X = X - mean
        # Get squared variance
        var2 = var**2
        # Calculate log likelihood
        log_likeli = -(n/2) * np.log(var2 * 2 * np.pi) - np.dot(adjusted_X, adjusted_X) / (2 * var2)

        return -log_likeli

    # Calculate initial guess for mean and standard deviation
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess]

    # Perform optimization through minimization of log likelihood
    result = minimize(lambda params: log_likelihood(params[0], params[1], X), initial_params)

    # Extract optimized parameters
    optimized_mean, optimized_std_dev = result.x
    
    # Print optimized mean and standard deviation
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)

    # Perform hypothesis test if specified
    if perform_hypothesis_test:
        # Calculate test statistic and p-value against standard normal (0, 1)
        test_statistic = (optimized_mean - 0) / optimized_std_dev  # Z-score
        p_value = 2 * (1 - norm.cdf(abs(test_statistic)))  # Two-tailed test

        # Determine if the null hypothesis (X is from a standard normal distribution) is rejected
        reject_null = p_value < 0.05  # Using a significance level of 0.05

        # Print hypothesis test results
        print("Test Statistic:", test_statistic)
        print("P-Value:", p_value)
        print("Reject Null Hypothesis:", reject_null)

    

    return optimized_mean, optimized_std_dev

def mle_t_distribution(X, y, perform_hypothesis_test=False):
    # Define the likelihood function for the t-distribution
    def log_likelihood(mean, var, df, X):
        adjusted_X = X - mean
        var2 = var**2
        log_likeli = np.sum(t.logpdf(adjusted_X / np.sqrt(var2), df))
        return -log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    df_guess = len(X)-1  # You can adjust the initial guess for degrees of freedom

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of log likelihood
    result = minimize(lambda params: log_likelihood(params[0], params[1], params[2], X), initial_params)

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    # Print optimized parameters
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)
    print("Optimized Degrees of Freedom:", optimized_df)

    # Perform hypothesis test if specified
    if perform_hypothesis_test:
        # Calculate test statistic and p-value against standard t-distribution (0, 1, df)
        test_statistic = (optimized_mean - 0) / (optimized_std_dev / np.sqrt(optimized_df))
        p_value = 2 * (1 - t.cdf(abs(test_statistic), df=optimized_df))  # Two-tailed test

        # Determine if the null hypothesis (X is from a standard t-distribution) is rejected
        reject_null = p_value < 0.05  # Using a significance level of 0.05

        # Print hypothesis test results
        print("Test Statistic:", test_statistic)
        print("P-Value:", p_value)
        print("Reject Null Hypothesis:", reject_null)

def mle_normal_distribution_one_input(X):
    # Define the likelihood function for the normal distribution
    def log_likelihood(params, X):
        mean, var = params
        n = len(X)
        # Adjust X to be centered around the mean
        adjusted_X = X - mean
        # Get squared variance
        var2 = var**2
        # Calculate log likelihood
        log_likeli = -(n/2) * np.log(var2 * 2 * np.pi) - np.sum(adjusted_X**2) / (2 * var2)

        return -log_likeli

    # Calculate initial guess for mean and standard deviation
    mean_guess = np.mean(X, axis=0)
    std_dev_guess = np.std(X, axis=0)
    
    # Initial guess for optimization as a 1D array
    initial_params = np.array([mean_guess, std_dev_guess])

    # Perform optimization through minimization of log likelihood
    result = minimize(log_likelihood, initial_params, args=(X,))

    # Extract optimized parameters
    optimized_mean, optimized_std_dev = result.x

    # Print optimized mean and standard deviation
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)

    return optimized_mean, optimized_std_dev

# Fixed MLE for t distribution for just the dataframe values input
def mle_t_distribution_one_input(y):
    # Define the likelihood function for the t-distribution
    def neg_log_likelihood(params, y):
        mean, var, df = params
        adjusted_y = y - mean
        log_likeli = -np.sum(t.logpdf(adjusted_y, df, loc=mean, scale=var))
        return log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    _, _, df_guess = t.fit(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of negative log likelihood
    result = minimize(neg_log_likelihood, initial_params, args=(y,), method='Nelder-Mead')

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    return optimized_mean, optimized_std_dev, optimized_df

def simulate_MA(N, num_steps, e, burn_in, mean, plot_y = False, max_threshold = 1e4):

    # Initialize y MA preds
    y = np.empty(num_steps)

    # Simulate the MA(N) process
    for i in range(1, num_steps + burn_in):
        y_t = mean + np.sum([0.05 * e[i - j] for j in range(1, N + 1)]) + e[i]
        if i > burn_in:
            # Check if y_t is beyond a certain threshold
            if abs(y_t) > max_threshold:
                y_t = np.sign(y_t) * mean
            y[i - burn_in - 1] = y_t

    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y)
    var_y = np.var(y)
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    if plot_y == True:
        # Plot the time series
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"MA({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.savefig(f'plots/MA_{N}_Steps.png')
        plt.show()


    return y, mean_y, var_y

def simulate_AR(N, num_steps, e, burn_in, mean, plot_y=True):
    # Initialize variables
    n = num_steps
    y = np.empty(n)

    # Simulate the AR(N) process
    for i in range(n + burn_in):
        y_t = mean  # Initialize y_t to the mean

        # Compute the AR(N) value for y_t
        for j in range(1, N + 1):
            if i - j >= 0:
                y_t += 0.5 ** j * y[i - j - burn_in] # take a look at removing the burn in

        # Add the white noise
        y_t += e[i]

        # Store the value in the y array if not in the burn-in period
        if i >= burn_in:
            y[i - burn_in] = y_t

    # Optionally plot the time series
    if plot_y:
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"AR({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.savefig(f'plots/AR_{N}_Steps.png')
        plt.show()


    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y[burn_in:])
    var_y = np.var(y[burn_in:])
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    return y, mean_y, var_y

def plot_acf_pacf(y, N, plot_type='AR', save_plots=False):
    # Set custom styling for the plots
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['xtick.color'] = 'red'
    plt.rcParams['ytick.color'] = 'red'
    plt.rcParams['text.color'] = 'white'

    # Create a directory to save plots if it doesn't exist
    if save_plots:
        import os
        if not os.path.exists('plots'):
            os.makedirs('plots')

    # Plot the ACF and PACF with red lines
    plt.figure(figsize=(12, 6))

    # ACF plot
    ax1 = plt.subplot(121)
    plot_acf(y, lags=40, ax=ax1, color='red')
    ax1.set_title("Autocorrelation Function (ACF)")

    # PACF plot
    ax2 = plt.subplot(122)
    plot_pacf(y, lags=40, ax=ax2, color='red')
    ax2.set_title("Partial Autocorrelation Function (PACF)")

    # Add an overall title including the plot_type
    plt.suptitle(f"{plot_type}({N}) - ACF and PACF Plots", color='white', fontsize=16)

    plt.savefig(f'plots/{plot_type}_{N}_ACF_PACF.png')

    plt.tight_layout()

    # Display the plots
    plt.show()


# Calculate VaR Normal Distribution:
def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, loc=mean, scale=std_dev)

    return -VaR

# Calculte VaR T Distribution:
def calc_var_t_dist(mean, std_dev, df, alpha=0.05):
    VaR = t.ppf(q=alpha, df=df, loc=mean, scale=std_dev)

    return -VaR

# Calculate ES for Normal
def calc_expected_shortfall_normal(mean, std_dev, alpha=0.05):
    
    # Calculate ES using the formula
    es = -1*mean + (std_dev * norm.pdf(norm.ppf(alpha, mean, std_dev)) / alpha)

    return es

# Calculate ES for Generalized T
def calc_expected_shortfall_t(mean, std_dev, df, alpha=0.05):
    # VaR for t dist
    var = -1*calc_var_t_dist(mean, std_dev, df, alpha=alpha)

    # PDF fucntion for t dist
    def t_pdf(x):
        return t.pdf(x, df, loc=mean, scale=std_dev)

    # Integrand for es
    def integrand(x):
        return x*t_pdf(x)

    # Calc ES using integration
    es, _ = quad(integrand, float("-inf"), var)

    return es/alpha

def calculate_ewma_covariance_matrix(df, lambd):
    # Calculate exponentially weighted covariance matrix provided a dataframe and lambda

    # Get the number of time steps n and vars m
    n, m = df.shape  
    
    # Initialize the exponentially weighted covariance matrix as a square matrix with dimensions (m, m)
    ewma_cov_matrix = np.zeros((m, m))  
    
    # Calculate the weights and normalized weights for each time step
    # w_{t_i} = (1-lambda)*lambda^{i-1}
    weights = [(1 - lambd) * lambd**(i) for i in range(n)]  
    weights = weights[::-1]
    #### Flip the weights

    # Calculate the sum of weights to normalize them
    total_weight = sum(weights)  # sum w_{t-j}
    
    # Normalize the weights by dividing each weight by the total weight
    # w_{t_i}^hat = w_{t_i} / sum w_{t-j}
    normalized_weights = [w / total_weight for w in weights]  
    
    # Calculate the means for each variable across all time steps
    means = df.mean(axis=0)  
    
    # Calculate the exponentially weighted covariance matrix
    for t in range(n):
        # Calculate the deviation of each variable at time t from its mean
        deviation = df.iloc[t, :] - means  
        
        # weighted deviation from means for x and y
        ewma_cov_matrix += normalized_weights[t] * deviation.values.reshape(-1, 1) @ deviation.values.reshape(1, -1)
    ewma_cov_matrix = pd.DataFrame(ewma_cov_matrix)
    return ewma_cov_matrix

def calculate_ewma_correlation_matrix(df, lambda_corr, lambda_cov=None):
    # Calculate exponentially weighted covariance matrix provided a dataframe and lambda
    
    if lambda_cov is None:
        lambda_cov = lambda_corr

    ewma_cov_matrix = calculate_ewma_covariance_matrix(df, lambda_cov)
    
    # Calculate the standard deviations for each variable across all time steps
    std_devs = np.sqrt(np.diag(ewma_cov_matrix))
    
    # Calculate the exponentially weighted correlation matrix
    ewma_corr_matrix = ewma_cov_matrix / np.outer(std_devs, std_devs)
        
    return pd.DataFrame(ewma_corr_matrix)

def chol_psd(a):
    if isinstance(a, pd.DataFrame):
        a = a.to_numpy()

    n = a.shape[0]

    # Initialize the root matrix with 0 values
    root = np.zeros((n, n), dtype=np.float64)
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
    # Ensure that a given matrix is almost positive semi-definite (PSD).
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

def higham_near_psd_np_array(a, epsilon = 0.0, max_iterations=100):
    # use Higham to ensure near psd matrix of a numpy array

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
        ### change Y to Covar matrix
        diffY = np.linalg.norm(Y - Yold, 'fro') / np.linalg.norm(Y, 'fro') #lambda calc

    return X

def higham_near_psd_dataframe(df, epsilon = 0.0, max_iterations=100):
    # use Higham to ensure near psd matrix of a numpy array
    a = df.to_numpy()
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
        ### change Y to Covar matrix
        diffY = np.linalg.norm(Y - Yold, 'fro') / np.linalg.norm(Y, 'fro') #lambda calc

    return X


# Implement a multivariate normal simulation that allows for simulation directly from a covar matrix or using PCA and parameter for % var explained
def multivariate_normal_simulation(mean, cov_matrix, num_samples, method='Direct', pca_explained_var=None):
    if method == 'Direct':
        cov_matrix = cov_matrix.values
        n = cov_matrix.shape[1]
        # Initialize an array to store the simulation results
        simulations = np.zeros((num_samples, n))

        L = chol_psd(cov_matrix)

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

def calculate_portfolio_var(portfolio, price_df, returns_df, lambd, alpha = 0.05):
    # calculate total portfolio value
    portfolio_value = 0.0
    # create array to store each stock's value
    delta = []
    for _, row in portfolio.iterrows():
        stock_value = row['Holding']*price_df[row['Stock']].iloc[-1]
        portfolio_value += stock_value
        delta.append(stock_value)

    print(f"Portfolio Value: {portfolio_value}")
    delta = np.array(delta)
    normalized_delta = delta / portfolio_value
    
    exp_weighted_cov = calculate_ewma_covariance_matrix(returns_df, lambd)
    exp_weighted_std = np.sqrt(np.diagonal(exp_weighted_cov))
    
    # Create a dictionary to store column titles and corresponding exp_weighted_std values
    result_dict = {column: std for column, std in zip(returns_df.columns, exp_weighted_std)}
    
    exp_weighted_std_portfolio = np.array([result_dict[stock] for stock in portfolio['Stock']])

    p_sig = np.sqrt(np.dot(np.dot(normalized_delta, exp_weighted_std_portfolio), normalized_delta))
    
    VaR = -delta * norm.ppf(1-alpha)*p_sig
    total_VaR = sum(VaR)

    print(f"Porftolio Value at Risk: ${total_VaR}\n")
    return total_VaR

def return_calculate(prices_df, method="DISCRETE", date_column="Date"):
    vars = prices_df.columns
    n_vars = len(vars)
    vars = [var for var in vars if var != date_column]
    
    if n_vars == len(vars):
        raise ValueError(f"date_column: {date_column} not in DataFrame: {vars}")
    
    n_vars = n_vars - 1
    
    p = prices_df[vars].values
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n-1, m))
    
    for i in range(n-1):
        for j in range(m):
            p2[i, j] = p[i+1, j] / p[i, j]
    
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    
    dates = prices_df[date_column].iloc[1:]
    
    # Create a new DataFrame with all columns
    data = {date_column: dates}
    for i in range(n_vars):
        data[vars[i]] = p2[:, i]
    
    out = pd.DataFrame(data)
    
    return out

def portfolio_es(portfolio, stock_dict, dist = "T"):
    if dist == "T":
        portfolio_es_individual = []
        for stock in portfolio['Stock']:
            mean = stock_dict[stock]['mean']
            std_dev = stock_dict[stock]['std_dev']
            df = stock_dict[stock]['df']
            stock_es = calc_expected_shortfall_t(mean, std_dev, df, alpha=0.05)
            stock_es *= portfolio.loc[portfolio['Stock'] == stock, 'Holding']
            portfolio_es_individual.append(stock_es)
        return np.mean(portfolio_es_individual)

def calculate_prices(returns, initial_price, method="classical_brownian", print_calc = True):
    #initial price
    prices = [initial_price]

    for i in range(len(returns)):
        r_t = returns.iloc[i]

        if method == "classical_brownian":
            # Classical Brownian Motion: P_t = P_{t-1} + r_t
            p_t = prices[i] + r_t
        elif method == "arithmetic_return":
            # Arithmetic Return System: P_t = P_{t-1}(r_t + 1)
            p_t = prices[i] * (1 + r_t)
        elif method == "geometric_brownian":
            # Log Return or Geometric Brownian Motion: P_t = P_{t-1}*e^{r_t}
            p_t = prices[i] * np.exp(r_t)
        else:
            raise ValueError("Invalid method. Supported methods are 'classical_brownian', 'arithmetic_return', and 'geometric_brownian'.")

        prices.append(p_t)

    expected_value = np.mean(prices)
    std_deviation = np.std(prices)
    if print_calc == True:
        print(f"Expected value of {method}: {expected_value}")
        print(f"Standard Deviation of {method}: {std_deviation}\n")

    return prices, expected_value, std_deviation


def integral_bsm_with_coupons(call, underlying, strike, days, rf, ivol, tradingDayYear, couponRate, function_type = "Black Scholes", q=None):
    
    if function_type == "Black Scholes":
        b = rf
    if function_type == "Merton":
        b = rf - q
    
    # time to maturity
    ttm = days / tradingDayYear

    # daily volatility with continuously compounded implied volatility
    dailyVol = ivol / np.sqrt(tradingDayYear)

    # std dev and mean for log normal distribution
    sigma = np.sqrt(days) * dailyVol
    mu = np.log(underlying) + ttm * b - 0.5 * sigma**2

    # log normal distribution
    d = lognorm(scale=np.exp(mu), s=sigma)

    # calculate the present value of coupons
    couponPV = 0.0
    for day in range(int(ttm * tradingDayYear)):
        # present value of the coupon payment for each day, 
         couponPV += couponRate * np.exp(-rf * (day / tradingDayYear))

    if call:
        # option value for call
        def f(x):
            return (max(0, x - strike) + couponPV) * d.pdf(x)
        val, _ = quad(f, 0, underlying * 2)
    else:
        # option value for put
        def g(x):
            return (max(0, strike - x) + couponPV) * d.pdf(x)
        val, _ = quad(g, 0, underlying * 2)

    return val * np.exp(-rf * ttm)


# Calculate options price
def options_price(S, X, T, sigma, r, b, option_type='call'):
    """
    S: Underlying Price
    X: Strike
    T: Time to Maturity(in years)
    sigma: implied volatility
    r: risk free rate
    b: cost of carry -> r if black scholes, r-q if merton
    """
    d1 = (math.log(S/X) + (b + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * math.exp((b - r) * T) * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)
    else:
        return X * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp((b - r) * T) * norm.cdf(-d1)
    

# function to calculate the option price error
def option_price_error(sigma, S, X, T, r, b, option_type, market_price):
    option_price = options_price(S, X, T, sigma, r, b, option_type)
    return abs(option_price - market_price)

# AR(1) method
def simulate_ar1_process(N, alpha, sigma, mu, num_steps):
    # Initialize variables
    y = np.empty((N, num_steps))
    
    for i in range(N):
        # Generate random noise
        epsilon = np.random.normal(0, sigma, num_steps)
        # Initialize the process with a random value
        y[i, 0] = mu + epsilon[0]
        
        for t in range(1, num_steps):
            y[i, t] = mu + alpha * (y[i, t - 1] - mu) + epsilon[t]
    
    return y


# Calculate implied volatility using bisection
def calculate_implied_volatility(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type, tol=1e-4, max_iter=300):
    S = curr_stock_price
    X = strike_price
    T = (options_expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuously_compounding_coupon
    b = r-q
    def calc_option_price(sigma):
        option_price = options_price(S, X, T, sigma, r, b, option_type)
        return option_price
    
    iteration = 0
    lower_vol = 0.001
    upper_vol = 15.0

    while iteration <= max_iter:
        mid_vol = (lower_vol + upper_vol) / 2
        option_price = calc_option_price(mid_vol)

        if abs(option_price) < tol:
            return mid_vol
        
        if option_price >0:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol

        iteration +=1

    raise ValueError( "Implied volatility calculation did not converge")

# separate implied volatility function to help puts converge
def calculate_implied_volatility_newton(curr_stock_price, strike_price, current_date, options_expiration_date, risk_free_rate, continuously_compounding_coupon, option_type, tol=1e-5, max_iter=500):
    S = curr_stock_price
    X = strike_price
    T = (options_expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuously_compounding_coupon
    b = r - q

    def calc_option_price(sigma):
        option_price = options_price(S, X, T, sigma, r, b, option_type)
        return option_price

    def calc_vega(sigma):
        d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp((b - r) * T) * norm.pdf(d1) * math.sqrt(T)
        return vega

    iteration = 0
    volatility = 0.2  # Initial guess

    while iteration <= max_iter:
        option_price = calc_option_price(volatility)
        vega = calc_vega(volatility)

        if abs(option_price) < tol:
            return volatility

        volatility = volatility - option_price / vega

        iteration += 1

    raise ValueError("Implied volatility calculation did not converge")

# Closed form greeks
def greeks(underlying_price, strike_price, risk_free_rate, implied_volatility, continuous_dividend_rate, current_date, expiration_date, option_type):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    d1 = (np.log(S/X)+(b+(0.5*sigma**2))*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        delta = np.exp((b-r)*T)*norm.cdf(d1)
        theta = -1*(S*np.exp((b-r)*T)*norm.pdf(d1)*sigma)/(0.5*np.sqrt(T)) - (b-r)*S*np.exp((b-r)*T)*norm.cdf(d1) - r*X*np.exp(-r*T)*norm.cdf(d2)
        rho = T*X*np.exp(-r*T)*norm.cdf(d2)
        carry_rho = T*S*np.exp((b-r)*T)*norm.cdf(d2)

    else:
        delta = np.exp((b-r)*T)*(norm.cdf(d1) - 1)
        theta = -1*(S*np.exp((b-r)*T)*norm.pdf(d1)*sigma)/(0.5*np.sqrt(T)) + (b-r)*S*np.exp((b-r)*T)*norm.cdf(-1*d1) + r*X*np.exp(-r*T)*norm.cdf(-1*d2)
        rho = -1*T*X*np.exp(-r*T)*norm.cdf(-d2)
        carry_rho = -1*T*S*np.exp((b-r)*T)*norm.cdf(-d2)

    gamma = norm.pdf(d1)*np.exp((b-r)*T)/(S*sigma*(np.sqrt(T)))
    vega = S*np.exp((b-r)*T)*norm.pdf(d1)*np.sqrt(T)

    return delta, gamma, vega, theta, rho, carry_rho

# finite difference derivative calculation greeks
def greeks_df(underlying_price, strike_price, risk_free_rate, implied_volatility, continuous_dividend_rate, current_date, expiration_date, option_type, epsilon = 0.01):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    #options_price(S, X, T, sigma, r, b, option_type='call')

    def derivative(variable = None):
        if variable == "underlying": # delta
            up_price = options_price(S+epsilon, X, T, sigma, r, b, option_type)
            down_price = options_price(S-epsilon, X, T, sigma, r, b, option_type)
            return (up_price-down_price)/(2*epsilon)
        if variable == "double_underlying": # gamma
            up_price = options_price(S+epsilon, X, T, sigma, r, b, option_type)
            down_price = options_price(S-epsilon, X, T, sigma, r, b, option_type)
            reg_price = options_price(S, X, T, sigma, r, b, option_type)
            return (up_price+down_price-2*reg_price)/(epsilon**2)
        if variable == "implied_volatility": # vega
            up_price = options_price(S, X, T, sigma+epsilon, r, b, option_type)
            down_price = options_price(S, X, T, sigma-epsilon, r, b, option_type)
            return (up_price-down_price)/(2*epsilon)
        if variable == "time_to_maturity": # theta
            up_price = options_price(S, X, T+epsilon, sigma, r, b, option_type)
            down_price = options_price(S, X, T-epsilon, sigma, r, b, option_type)
            return -(up_price-down_price)/ (2*epsilon)
        if variable == 'risk_free_rate': #rho
            up_price = options_price(S, X, T, sigma, r+epsilon, b, option_type)
            down_price = options_price(S, X, T, sigma, r-epsilon, b, option_type)
            return (up_price-down_price)/(2*epsilon)
        if variable == 'cost_of_carry': # carry rho
            up_price = options_price(S, X, T, sigma, r, b+epsilon, option_type)
            down_price = options_price(S, X, T, sigma, r, b-epsilon, option_type)
            return (up_price-down_price)/(2*epsilon)

    delta = derivative("underlying")
    gamma = derivative("double_underlying")
    vega = derivative("implied_volatility")
    theta = derivative("time_to_maturity")
    rho = derivative("risk_free_rate")
    carry_rho = derivative("cost_of_carry")

    return delta, gamma, vega, theta, rho, carry_rho

def greeks_with_dividends(underlying_price, strike_price, risk_free_rate, implied_volatility, continuous_dividend_rate, current_date, expiration_date, option_type, div_dates, div_amounts):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    # Calculate present value of dividends
    pv_dividends = 0
    for div_date, div_amount in zip(div_dates, div_amounts):
        if div_date > current_date and div_date < expiration_date:
            pv_dividends += div_amount * np.exp(-r * (div_date - current_date).days / 365)

    # Adjust underlying price for dividends
    S_adj = S - pv_dividends

    d1 = (np.log(S_adj / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = np.exp((b - r) * T) * norm.cdf(d1)
        theta = -1 * (S_adj * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T)) - (b - r) * S_adj * np.exp((b - r) * T) * norm.cdf(d1) - r * X * np.exp(-r * T) * norm.cdf(d2)
        rho = T * X * np.exp(-r * T) * norm.cdf(d2)
        carry_rho = T * S_adj * np.exp((b - r) * T) * norm.cdf(d2)

    else:
        delta = np.exp((b - r) * T) * (norm.cdf(d1) - 1)
        theta = -1 * (S_adj * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T)) + (b - r) * S_adj * np.exp((b - r) * T) * norm.cdf(-1 * d1) + r * X * np.exp(-r * T) * norm.cdf(-1 * d2)
        rho = -1 * T * X * np.exp(-r * T) * norm.cdf(-d2)
        carry_rho = -1 * T * S_adj * np.exp((b - r) * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) * np.exp((b - r) * T) / (S_adj * sigma * (np.sqrt(T)))
    vega = S_adj * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, vega, theta, rho, carry_rho


# Binomial tree European option
def binomial_tree_option_pricing_european(underlying_price, strike_price, current_date, expiration_date, risk_free_rate, dividend_yield, implied_volatility, num_steps, option_type):
    S = underlying_price
    X = strike_price
    T = (expiration_date - current_date).days / 365.0
    r = risk_free_rate
    q = dividend_yield
    b = r - q
    sigma = implied_volatility
    N = num_steps

    # parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp((b) * dt) - d) / (u - d)
    pd = 1.0 - pu

    # initialize arrays
    ps = np.zeros(N+1)
    paths = np.zeros(N+1)
    prices = np.zeros(N+1)

    # calculate factorials
    n_fact = math.factorial(N)

    # calculate stock prices at each node
    for i in range(N+1):
        prices[i] = S * u**i * d**(N-i)
        ps[i] = pu**i * pd**(N-i)
        paths[i] = n_fact / (math.factorial(i) * math.factorial(N-i))

    # Calculate option payoff at each leaf
    if option_type == 'call':
        prices = np.maximum(0, prices-X)
    else:
        prices = np.maximum(X-prices, 0)

    # calculate final option prices as the discounted expected payoff
    prices = prices * ps
    option_price = np.dot(prices, paths)
    return np.exp(-r * T) * option_price


def binomial_tree_option_pricing_american(underlying_price, strike_price, ttm, risk_free_rate, b, implied_volatility, num_steps, option_type):
    S = underlying_price
    X = strike_price
    T = ttm
    r = risk_free_rate
    sigma = implied_volatility
    N = num_steps

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    if option_type == 'call':
        z = 1
    else:
        z = -1
    
    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i
    
    nNodes = nNodeFunc(N) - 1

    optionValues = np.empty(nNodes+1)  # Increase the size by 1

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = S * (u ** i) * (d ** (j - i))
            optionValues[idx] = max(0, z * (price - X))

            if j < N:
                optionValues[idx] = max(
                    optionValues[idx],
                    df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]),
                )

    return optionValues[0]

def binomial_tree_option_pricing_american_complete(underlying_price, strike_price, ttm, risk_free_rate, implied_volatility, num_steps, option_type, div_amounts = None, div_times = None):
    S = underlying_price
    X = strike_price
    T = ttm
    r = risk_free_rate
    sigma = implied_volatility
    N = num_steps

    if (div_amounts is None) or (div_times is None) or len(div_amounts) == 0 or len(div_times) == 0 or div_times[0] > N:
        return binomial_tree_option_pricing_american(S, X, T, risk_free_rate, risk_free_rate, implied_volatility, num_steps, option_type)
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    if option_type == 'call':
        z = 1
    else:
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i + 1

    nDiv = len(div_times)
    n_nodes = nNodeFunc(N)
    option_values = np.empty(n_nodes + 1)  # Increase the size by 1

    for j in range(div_times[0], -1, -1):  # Use a float range for j
        for i in range(j, -1, -1):  # Use a float range for i
            idx = idxFunc(i, j)
            price = S * (u ** i) * (d ** (j - i))

            if j < div_times[0]:
                # times before or at the dividend working backward induction
                option_values[idx] = max(0, z * (price - X))
                option_values[idx] = max(option_values[idx], df * (pu * option_values[idxFunc(i + 1, j + 1)] + pd * option_values[idxFunc(i, j + 1)]))
            else:
                # time after the dividend
                val_no_exercise = binomial_tree_option_pricing_american_complete(price-div_amounts[0], X, ttm-div_times[0]*dt, risk_free_rate, implied_volatility, N-div_times[0], option_type, div_amounts[1:nDiv], div_times[1:nDiv] - div_times[0])
                val_exercise = max(0, z * (price - X))
                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[0]

# Function to calculate the portfolio value for a given underlying value
def calculate_portfolio_value_american(underlying_value, portfolio, current_date, dividend_payment_date, risk_free_rate):
    portfolio_value = 0.0

    for _, asset in portfolio.iterrows():
        if asset['Type'] == 'Option':
            S = underlying_value
            X = asset['Strike']
            expiration_date = datetime.strptime(asset['ExpirationDate'], "%m/%d/%Y")
            T = (expiration_date - current_date).days / 365.0
            option_type = asset['OptionType']
            dividend_payment_time = np.array([(dividend_payment_date - current_date).days])
            dividend_payment = np.array([1])

            implied_volatility = calculate_implied_volatility_newton(S, X, current_date, expiration_date, risk_free_rate, 0, option_type)

            # Calculate the american option price using tree method
            option_value = binomial_tree_option_pricing_american_complete(S, X, T, risk_free_rate, implied_volatility, (dividend_payment_date - current_date).days+1, option_type, dividend_payment, dividend_payment_time)

            # Add or subtract option value to the portfolio based on Holding (1 or -1)
            portfolio_value += asset['Holding'] * option_value
        elif asset['Type'] == 'Stock':
            # If it's a stock, just add its current price to the portfolio value
            portfolio_value += asset['Holding'] * (asset['CurrentPrice'] - underlying_value)

    return portfolio_value