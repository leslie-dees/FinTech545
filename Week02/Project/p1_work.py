import sys
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')

from fin_package import first4Moments
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_1samp

# Create a standard normal distribution and set parameters for testing
mean = 0
std_dev = 1
known_kurtosis = 3
known_skew = 0
num_samples = 5000
sample_size = 5000

def calc_estimated_kurtosis(sample_size=sample_size):
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

# Calculate the average estimated kurtosis
averaged_estimated_kurtosis, estimated_kurtosis_vals, averaged_estimated_skew, estimated_skew_vals = calc_estimated_kurtosis()

# Perform a one-sample t-test
t_stat1, p_value1 = ttest_1samp(estimated_kurtosis_vals, known_kurtosis)

# Compare the average estimated kurtosis to the known kurtosis
print(f"Known Kurtosis: {known_kurtosis}")
print(f"Average Estimated Kurtosis (Sample Size {sample_size}): {averaged_estimated_kurtosis}")

# Hypothesis test results
print(f"T-statistic: {t_stat1}")
print(f"P-value: {p_value1}")

# Check if the result is statistically significant (e.g., using a significance level of 0.05)
if p_value1 < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")

# Perform a one-sample t-test
t_stat2, p_value2 = ttest_1samp(estimated_skew_vals, known_skew)

# Compare the average estimated kurtosis to the known kurtosis
print(f"Known Skew: {known_skew}")
print(f"Average Estimated Skew (Sample Size {sample_size}): {averaged_estimated_skew}")

# Hypothesis test results
print(f"T-statistic: {t_stat2}")
print(f"P-value: {p_value2}")

# Check if the result is statistically significant (e.g., using a significance level of 0.05)
if p_value1 < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
