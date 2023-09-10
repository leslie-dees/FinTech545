from scipy.stats import moment
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
from scipy.optimize import minimize

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
        sns.histplot(error_vector, kde=True, color='blue', bins=30)
        plt.title("Error Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()
    
    print("Error Vector: ", error_vector)
    averaged_error_vector = np.mean(error_vector)
    print("Averaged Error Vector: ", averaged_error_vector)
    return error_vector

def optimize_normal_distribution(X, y, perform_hypothesis_test = False):
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

def optimize_t_distribution(X, y, perform_hypothesis_test=False):
    # Define the likelihood function for the t-distribution
    def log_likelihood(mean, var, df, X):
        adjusted_X = X - mean
        var2 = var**2
        log_likeli = np.sum(t.logpdf(adjusted_X / np.sqrt(var2), df))
        return -log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    df_guess = 3.0  # You can adjust the initial guess for degrees of freedom

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


def simulate_MA(N, num_steps, e, burn_in, mean, plot_y = False, max_threshold = 1e6):

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
    mean_y = np.mean(y[burn_in:])
    var_y = np.var(y[burn_in:])
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    if plot_y == True:
        # Plot the time series
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"MA({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.show()

    return y, mean_y, var_y

def simulate_AR(N, num_steps, e, burn_in, mean, plot_y = False):
    
    # Initialize y MA preds and AR coefficients
    y = np.empty(num_steps)
    phi = np.random.uniform(-1, 1, size=N)

    # Loop through time steps
    for i in range(num_steps + burn_in):
        # If we have enough data for AR(N) process
        if i >= N:
            y_t = mean + np.sum(phi * y[i - N:i]) + e[i]
        else:
            # Use white noise if not enough lagged values    
            y_t = mean + e[i]

        if i >= burn_in:
            y[i - burn_in] = y_t

    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y[burn_in:])
    var_y = np.var(y[burn_in:])
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    # Optionally plot the time series if plot_y is True
    if plot_y:
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"AR({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.show()

    # Return the simulated time series, mean, and variance
    return y, mean_y, var_y

def simulate_AR(N, num_steps, e, burn_in, mean, plot_y=True):
    # Initialize variables
    n = num_steps
    y = np.empty(n)

    # Simulate the AR(N) process
    for i in range(n + burn_in):
        y_t = mean  # Initialize y_t to zero

        # Compute the AR(N) value for y_t
        for j in range(1, N + 1):
            if i - j >= 0:
                y_t += 0.5 ** j * y[i - j - burn_in]

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
        plt.show()

    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y[burn_in:])
    var_y = np.var(y[burn_in:])
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    return y, mean_y, var_y