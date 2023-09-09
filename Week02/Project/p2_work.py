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
num_samples = 1000
sample_size = 5000

def calc_estimated_kurtosis(sample_size=sample_size):
    # Array to keep kurtosis values for 
    estimated_kurtosis_vals = []

    # Test a new kurtosis for the number of sample distributions available
    for _ in tqdm(range(num_samples), desc="Generating Samples"):
        # Create random normal sample distribution
        sample = np.random.normal(mean, std_dev, sample_size)

        # Calculate kurtosis using your function (first4Moments)
        _, _, _, kurtosis = first4Moments(sample, excess_kurtosis=False)

        estimated_kurtosis_vals.append(kurtosis)

    # Average the estimated kurtosis using your function      
    averaged_estimated_kurtosis = np.mean(estimated_kurtosis_vals)
    return averaged_estimated_kurtosis, estimated_kurtosis_vals

# Calculate the average estimated kurtosis
averaged_estimated_kurtosis, estimated_kurtosis_vals = calc_estimated_kurtosis()

# Perform a one-sample t-test
t_stat, p_value = ttest_1samp(estimated_kurtosis_vals, known_kurtosis)

# Compare the average estimated kurtosis to the known kurtosis
print(f"Known Kurtosis: {known_kurtosis}")
print(f"Average Estimated Kurtosis (Sample Size {sample_size}): {averaged_estimated_kurtosis}")

# Hypothesis test results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Check if the result is statistically significant (e.g., using a significance level of 0.05)
if p_value < 0.05:
    print("The difference is statistically significant.")
else:
    print("The difference is not statistically significant.")
